"""
inference.py — 推理与文本生成脚本

本脚本负责加载训练好的 Transformer 模型并生成文本。
支持两种生成策略：
1. 贪心搜索（Greedy Search）：每步选择概率最高的 token
2. 温度采样（Temperature Sampling）：按温度调节后的概率分布随机采样

使用方法：
    # 使用默认种子文本生成
    python -m src.inference --model experiments/baseline/model_best.pt

    # 指定种子文本
    python -m src.inference --model experiments/baseline/model_best.pt --seed "ROMEO:"

    # 指定生成长度和温度
    python -m src.inference --model experiments/baseline/model_best.pt --length 500 --temperature 0.8

    # 交互模式
    python -m src.inference --model experiments/baseline/model_best.pt --interactive
"""

import os
import sys
import argparse
import torch

# 确保可以导入本项目的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.dataset import CharTokenizer, PAD_ID, BOS_ID, EOS_ID


def load_model(model_path, device):
    """
    加载训练好的模型。

    参数:
        model_path (str): 模型文件路径（.pt 文件）。
        device (torch.device): 加载到的设备。

    返回:
        tuple: (model, tokenizer, config)
            - model: 加载权重后的 Transformer 模型
            - tokenizer: CharTokenizer 实例
            - config: 训练时使用的配置
    """
    print(f"加载模型: {model_path}")

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # 加载分词器
    data_dir = config["data"]["data_dir"]

    # 确定项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(project_root, data_dir, "vocab.json")

    tokenizer = CharTokenizer()
    tokenizer.load(vocab_path)
    print(f"词表大小: {tokenizer.vocab_size}")

    # 构建模型
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        num_layers=config["model"]["num_layers"],
        max_len=config["model"]["max_len"],
        dropout=config["model"]["dropout"],
        pad_id=PAD_ID,
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    val_loss = checkpoint.get("val_loss", "unknown")
    print(f"模型来自 Epoch {epoch}, 验证 Loss: {val_loss}")

    return model, tokenizer, config


def generate_greedy(model, tokenizer, device, seed_text, max_len=200):
    """
    使用贪心搜索生成文本。

    贪心搜索策略：每一步都选择概率最高的 token。
    优点：生成结果确定性强，质量通常不错。
    缺点：可能缺乏多样性，容易陷入重复。

    参数:
        model: 训练好的 Transformer 模型。
        tokenizer: 字符分词器。
        device: 计算设备。
        seed_text (str): 种子文本（作为源序列输入编码器）。
        max_len (int): 最大生成长度。

    返回:
        str: 生成的文本。
    """
    model.eval()

    with torch.no_grad():
        # 编码源序列
        src_ids = tokenizer.encode(seed_text)
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        # src: [1, src_len]

        # 初始化目标序列
        # 使用源序列的最后一个字符作为起始 token
        tgt_ids = [src_ids[-1]] if src_ids else [BOS_ID]

        for step in range(max_len):
            tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            # tgt: [1, current_len]

            # 前向传播
            # logits: [1, current_len, vocab_size]
            logits, _ = model(src, tgt)

            # 取最后一个时间步的预测
            # next_logits: [1, vocab_size]
            next_logits = logits[:, -1, :]

            # 贪心：选择概率最高的 token
            next_token = next_logits.argmax(dim=-1).item()

            # 添加到生成序列
            tgt_ids.append(next_token)

            # 遇到 EOS 就停止
            if next_token == EOS_ID:
                break

    # 解码并返回
    generated = tokenizer.decode(tgt_ids)
    return generated


def generate_with_temperature(model, tokenizer, device, seed_text,
                               max_len=200, temperature=0.8, top_k=0):
    """
    使用温度采样生成文本。

    温度参数控制生成的随机性：
    - temperature < 1.0：更保守，倾向于高概率 token（更确定）
    - temperature = 1.0：按原始概率分布采样
    - temperature > 1.0：更随机，增加低概率 token 的出现机会（更多样）

    参数:
        model: 训练好的 Transformer 模型。
        tokenizer: 字符分词器。
        device: 计算设备。
        seed_text (str): 种子文本。
        max_len (int): 最大生成长度。
        temperature (float): 温度参数（0.1 ~ 2.0）。
        top_k (int): Top-K 采样，0 表示不使用。

    返回:
        str: 生成的文本。
    """
    model.eval()

    with torch.no_grad():
        src_ids = tokenizer.encode(seed_text)
        src = torch.tensor([src_ids], dtype=torch.long).to(device)

        tgt_ids = [src_ids[-1]] if src_ids else [BOS_ID]

        for step in range(max_len):
            tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            logits, _ = model(src, tgt)
            next_logits = logits[:, -1, :]  # [1, vocab_size]

            # 温度缩放
            next_logits = next_logits / temperature

            # Top-K 过滤
            if top_k > 0:
                # 只保留前 K 个最大的 logit，其余设为 -inf
                values, indices = next_logits.topk(top_k)
                mask = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(1, indices, values)
                next_logits = mask

            # 转换为概率分布并采样
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            tgt_ids.append(next_token)

            if next_token == EOS_ID:
                break

    generated = tokenizer.decode(tgt_ids)
    return generated


def interactive_mode(model, tokenizer, device):
    """
    交互式文本生成模式。

    用户可以反复输入种子文本，模型生成续写内容。
    输入 'quit' 或 'exit' 退出。
    """
    print("\n" + "=" * 50)
    print("交互式文本生成模式")
    print("=" * 50)
    print("输入种子文本，模型将为你续写。")
    print("输入 'quit' 或 'exit' 退出。")
    print("输入 'temp=0.8' 可以调节温度参数。")
    print("=" * 50)

    temperature = 0.8
    max_len = 300

    while True:
        print()
        seed = input("📝 输入种子文本: ").strip()

        if not seed:
            continue
        if seed.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 检查是否在设置参数
        if seed.startswith("temp="):
            try:
                temperature = float(seed.split("=")[1])
                print(f"温度已设置为: {temperature}")
            except ValueError:
                print("无效的温度值！")
            continue
        if seed.startswith("len="):
            try:
                max_len = int(seed.split("=")[1])
                print(f"最大长度已设置为: {max_len}")
            except ValueError:
                print("无效的长度值！")
            continue

        print(f"\n🎭 生成结果 (temperature={temperature}):")
        print("─" * 50)

        # 贪心生成
        greedy_text = generate_greedy(model, tokenizer, device, seed, max_len=max_len)
        print("【贪心搜索】")
        print(greedy_text)

        print()

        # 温度采样生成
        sampled_text = generate_with_temperature(
            model, tokenizer, device, seed, max_len=max_len, temperature=temperature
        )
        print(f"【温度采样 T={temperature}】")
        print(sampled_text)

        print("─" * 50)


def main():
    """主推理函数。"""
    parser = argparse.ArgumentParser(description="Transformer 文本生成推理")
    parser.add_argument("--model", type=str, required=True,
                        help="模型文件路径（.pt）")
    parser.add_argument("--seed", type=str, default=None,
                        help="种子文本")
    parser.add_argument("--length", type=int, default=300,
                        help="生成长度（默认 300）")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度（默认 0.8）")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K 采样（默认 0，不使用）")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (cpu/cuda)")
    parser.add_argument("--interactive", action="store_true",
                        help="进入交互模式")
    args = parser.parse_args()

    # 确定设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model, tokenizer, config = load_model(args.model, device)

    # 交互模式
    if args.interactive:
        interactive_mode(model, tokenizer, device)
        return

    # 确定种子文本
    if args.seed:
        seed_text = args.seed
    else:
        # 使用数据集前 seq_len 个字符作为种子
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, config["data"]["data_dir"], "input.txt")
        with open(data_path, "r", encoding="utf-8") as f:
            seed_text = f.read()[:config["training"]["seq_len"]]

    print(f"\n种子文本: {repr(seed_text[:100])}")

    # 贪心搜索
    print("\n" + "=" * 50)
    print("【贪心搜索】")
    print("=" * 50)
    greedy_text = generate_greedy(model, tokenizer, device, seed_text, args.length)
    print(greedy_text)

    # 温度采样
    print("\n" + "=" * 50)
    print(f"【温度采样 T={args.temperature}】")
    print("=" * 50)
    sampled_text = generate_with_temperature(
        model, tokenizer, device, seed_text, args.length,
        temperature=args.temperature, top_k=args.top_k
    )
    print(sampled_text)


if __name__ == "__main__":
    main()
