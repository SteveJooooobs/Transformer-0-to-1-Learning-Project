# -*- coding: utf-8 -*-
"""
统一训练与评估模块：train.py
本模块是整个项目的运行入口，包含以下核心功能：
1. 命令行参数解析：支持切换模型版本（handwritten vs torch）、设置训练周期、批大小、学习率等。
2. 完整的训练与验证循环：计算交叉熵损失，评估字符级别（Token-level）准确率和句子级别（Sequence-level）整句完全匹配率。
3. 贪婪解码推理（Greedy Decoding）：在验证和测试时，逐步生成标准日期格式。
4. 双重注意力可视化：
   - 终端 ASCII 阴影字符画热力图：直接在命令行展示，极其直观且无需任何依赖。
   - Matplotlib 图像热力图：保存为高清 PNG 图片，适合报告展示（采用 try-except 优雅降级保护）。
"""

import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from models.handwritten_transformer import HandwrittenTransformer
from models.torch_transformer import TorchTransformer

# 尝试导入 matplotlib 以便绘制图像，若未安装则优雅降级
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 包含所有配置参数的对象。
    """
    parser = argparse.ArgumentParser(description="Transformer 训练与对比平台 - 日期格式转换任务")
    parser.add_argument(
        "--model",
        type=str,
        default="handwritten",
        choices=["handwritten", "torch"],
        help="选择运行的模型版本: 'handwritten' (自写版本) 或 'torch' (官方封装版)"
    )
    parser.add_argument("--epochs", type=int, default=5, help="训练周期数 (默认 5，在 CPU 上约需 1~2 分钟)")
    parser.add_argument("--batch_size", type=int, default=128, help="批处理大小 (默认 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率 (默认 0.001)")
    parser.add_argument("--num_samples", type=int, default=8000, help="数据集样本生成数量 (默认 8000)")
    parser.add_argument("--d_model", type=int, default=128, help="模型特征维度 (默认 128)")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数 (默认 4)")
    parser.add_argument("--d_ff", type=int, default=256, help="前馈网络隐藏层维度 (默认 256)")
    parser.add_argument("--num_layers", type=int, default=2, help="编码器/解码器层数 (默认 2)")
    parser.add_argument("--dropout", type=float, default=0.1, help="随机失活率 (默认 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    
    return parser.parse_args()


def set_seed(seed):
    """
    固定随机种子以保证实验的可复现性。
    
    Args:
        seed (int): 种子数值。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def greedy_decode(model, src, tokenizer, max_tgt_len=15):
    """
    使用贪婪解码 (Greedy Decoding) 进行推断预测。
    
    Args:
        model (nn.Module): 训练好的 Transformer 模型。
        src (Tensor): 输入的源序列编码张量，形状为 [B, L_src]。
        tokenizer (CharacterTokenizer): 分词器实例。
        max_tgt_len (int): 目标生成的最大长度。
        
    Returns:
        tuple: (预测的 Token ID 张量 [B, max_tgt_len], 最后一层交叉注意力权重 [B, L_tgt, L_src])
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device
    
    # 1. 初始化解码器输入，第一列全部填充为序列开始标记 <sos>
    ys = torch.full((batch_size, 1), tokenizer.sos_id, dtype=torch.long, device=device)
    
    last_attn = None
    
    # 2. 逐步自回归预测下一个 Token
    for i in range(max_tgt_len - 1):
        with torch.no_grad():
            # 进行前向传播
            # ys 的长度随着预测不断增加
            logits, attn = model(src, ys)
            
        # 取最后一个时间步的概率输出，并选择概率最大的 Token
        # logits 形状为 [B, cur_len, vocab_size]，我们取 logits[:, -1, :] -> [B, vocab_size]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # 拼接至解码器输入中
        ys = torch.cat([ys, next_token], dim=1)
        
        # 保存最后一步完整的交叉注意力权重（形状：[B, num_heads, cur_len, L_src] 或 [B, cur_len, L_src]）
        last_attn = attn
        
    return ys, last_attn


def get_average_attention(attn):
    """
    统一注意力矩阵维度：如果是 4 维张量 [B, H, L_tgt, L_src] (手写版多头注意力)，则在头维度 (H) 上求平均；
    如果是 3 维张量 [B, L_tgt, L_src] (官方版已求过平均)，则直接返回。
    
    Args:
        attn (Tensor): 注意力权重张量。
        
    Returns:
        Tensor: 统一为三维的注意力张量，形状为 [B, L_tgt, L_src]。
    """
    if attn.dim() == 4:
        return attn.mean(dim=1)
    return attn


def visualize_ascii_attention(src_text, pred_text, attn_matrix):
    """
    在终端打印极具视觉美感的 ASCII 阴影字符画热力图。
    使用常规 ASCII 字符（空格, ., -, =, #）避免 Windows 下的 GBK 编码报错。
    
    Args:
        src_text (str): 原始输入文本。
        pred_text (str): 模型预测文本。
        attn_matrix (Tensor): 对应的交叉注意力权重二维矩阵，形状为 [L_tgt, L_src]。
    """
    print("\n【控制台 ASCII 注意力对齐热力图】")
    print(" 字符表示： 空白(<5%), .(<20%), -(<50%), =(<80%), #(>=80% 强关注)")
    print("-" * 75)
    
    # 字符映射辅助函数
    def val_to_block(val):
        if val >= 0.8: return "#"
        if val >= 0.5: return "="
        if val >= 0.2: return "-"
        if val >= 0.05: return "."
        return " "

    # 包装特殊标记，使列标题与 Token 序列对齐
    src_tokens = ["<sos>"] + list(src_text) + ["<eos>"]
    pred_tokens = ["<sos>"] + list(pred_text) + ["<eos>"]
    
    # 限制绘制范围，防止维度不匹配越界
    num_rows = min(len(pred_tokens), attn_matrix.size(0))
    num_cols = min(len(src_tokens), attn_matrix.size(1))
    
    # 打印表头（输入字符）
    header = f"{'输出 \\ 输入':<12} | " + " ".join([f"{c:^3}" for c in src_tokens[:num_cols]])
    print(header)
    print("-" * len(header))
    
    # 逐行打印（输出字符及注意力分布）
    for i in range(num_rows):
        row_char = pred_tokens[i]
        blocks = []
        for j in range(num_cols):
            attn_val = attn_matrix[i, j].item()
            blocks.append(f"{val_to_block(attn_val):^3}")
        print(f"{row_char:<12} | " + " ".join(blocks))
    print("-" * 75 + "\n")


def save_matplotlib_heatmap(src_text, pred_text, attn_matrix, filename):
    """
    使用 Matplotlib 绘制并保存高清的注意力机制对齐热力图。
    
    Args:
        src_text (str): 原始输入文本。
        pred_text (str): 模型预测文本。
        attn_matrix (Tensor): 对应的交叉注意力权重二维矩阵，形状为 [L_tgt, L_src]。
        filename (str): 图像保存路径。
    """
    if not HAS_MATPLOTLIB:
        return
        
    src_tokens = ["<sos>"] + list(src_text) + ["<eos>"]
    pred_tokens = ["<sos>"] + list(pred_text) + ["<eos>"]
    
    # 裁剪张量对齐实际文本长度
    num_rows = min(len(pred_tokens), attn_matrix.size(0))
    num_cols = min(len(src_tokens), attn_matrix.size(1))
    
    attn_np = attn_matrix[:num_rows, :num_cols].cpu().numpy()
    
    # 绘图设置
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(attn_np, cmap="bone")
    fig.colorbar(cax)
    
    # 设置刻度标记
    ax.set_xticks(range(num_cols))
    ax.set_yticks(range(num_rows))
    ax.set_xticklabels(src_tokens[:num_cols], rotation=45, ha='left')
    ax.set_yticklabels(pred_tokens[:num_rows])
    
    # 设定主标题
    plt.title("Attention Alignment Heatmap", pad=20)
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"【成功】已将注意力机制高清热力图保存至: {filename}")


def evaluate(model, val_loader, tokenizer, device):
    """
    在验证集上评估模型性能。
    计算：
    1. 平均 CrossEntropy Loss。
    2. Token 级分类准确率（忽略填充字符）。
    3. Sequence 级整句完全匹配准确率（最符合直观结果的指标）。
    
    Args:
        model (nn.Module): 待评估模型。
        val_loader (DataLoader): 验证集加载器。
        tokenizer (CharacterTokenizer): 分词器实例。
        device (torch.device): 硬件运行设备。
        
    Returns:
        tuple: (平均 Loss, Token 准确率, 整句匹配率)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    
    total_seqs = 0
    correct_seqs = 0
    
    with torch.no_grad():
        for batch in val_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            # 1. 计算评估 Loss (逻辑同训练阶段)
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]
            
            logits, _ = model(src, dec_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
            total_loss += loss.item() * src.size(0)
            
            # 2. 计算 Token-level 准确率
            preds_token = logits.argmax(dim=-1) # [B, L_dec_input]
            active_mask = (dec_target != tokenizer.pad_id)
            correct_tokens += ((preds_token == dec_target) & active_mask).sum().item()
            total_tokens += active_mask.sum().item()
            
            # 3. 计算 Sequence-level (整句) 准确率，使用贪婪解码进行自回归完整生成
            decoded_ids, _ = greedy_decode(model, src, tokenizer, max_tgt_len=tgt.size(1))
            
            for i in range(src.size(0)):
                pred_str = tokenizer.decode(decoded_ids[i])
                target_str = batch["tgt_raw"][i]
                
                if pred_str == target_str:
                    correct_seqs += 1
                total_seqs += 1
                
    avg_loss = total_loss / len(val_loader.dataset)
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    seq_acc = correct_seqs / total_seqs if total_seqs > 0 else 0.0
    
    return avg_loss, token_acc, seq_acc


def main():
    """
    主控制流：配置参数、加载数据、初始化模型、执行训练循环、记录耗时并保存结果与可视化。
    """
    args = parse_args()
    set_seed(args.seed)
    
    # 确定硬件设备，CPU 友好型配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行硬件设备: {device}")
    
    # 1. 准备数据加载器与分词器
    print("正在生成模拟日期数据集...")
    train_loader, val_loader, tokenizer = get_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        val_ratio=0.2,
        max_src_len=30,
        max_tgt_len=15
    )
    print(f"数据集划分完成。训练集样本: {len(train_loader.dataset)}，验证集样本: {len(val_loader.dataset)}")
    print(f"词表构建完成，大小 (vocab_size): {tokenizer.vocab_size}")
    
    # 2. 实例化对应模型
    if args.model == "handwritten":
        print("\n==> 正在使用：【纯手写实现版本】 HandwrittenTransformer")
        model = HandwrittenTransformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            max_len=100,
            dropout=args.dropout
        ).to(device)
    else:
        print("\n==> 正在使用：【官方封装工具版本】 TorchTransformer")
        model = TorchTransformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            max_len=100,
            dropout=args.dropout
        ).to(device)
        
    # 3. 统计模型总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总量为: {total_params:,}")
    
    # 4. 配置优化器和损失准则 (忽略 Padding)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    # 5. 执行训练循环
    print(f"\n开始训练，共计 {args.epochs} 个 Epoch...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        epoch_start = time.time()
        
        for batch in train_loader:
            src = batch["src"].to(device)  # [B, L_src]
            tgt = batch["tgt"].to(device)  # [B, L_tgt]
            
            # 对目标序列做错位切片：
            # dec_input 输入给解码器，去掉最后一列 <eos> 符号 -> [B, L_tgt-1]
            # dec_target 供计算 CrossEntropyLoss 监督，去掉第一列 <sos> 符号 -> [B, L_tgt-1]
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]
            
            # 清空优化器梯度
            optimizer.zero_grad()
            
            # 前向传播计算 logits
            logits, _ = model(src, dec_input)
            
            # 展平维度计算 CrossEntropy
            # logits 变更为 [B * (L_tgt-1), vocab_size]，dec_target 变更为 [B * (L_tgt-1)]
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
            
            # 反向传播与参数更新
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        # 计算当前 Epoch 的训练与验证指标
        avg_train_loss = epoch_loss / batch_count
        val_loss, token_acc, seq_acc = evaluate(model, val_loader, tokenizer, device)
        epoch_duration = time.time() - epoch_start
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Token Acc: {token_acc * 100:.2f}% | Val Seq Acc (整句匹配率): {seq_acc * 100:.2f}%")
              
    total_duration = time.time() - start_time
    print("=" * 80)
    print(f"训练全部结束！总耗时: {total_duration:.1f} 秒，平均每个 Epoch 耗时: {total_duration / args.epochs:.1f} 秒。")
    
    # 6. 单例推断展示与注意力可视化 (选取几个典型样例查看)
    print("\n" + "#" * 35 + " 推理测试与可视化验证 " + "#" * 35)
    test_cases = [
        "May 23, 2026",
        "26/05/2026",
        "2026-05-23",
        "Saturday, May 23, 2026",
        "23-05-26",
        "23.05.2026",
        "May 23 26"
    ]
    
    model.eval()
    for idx, raw_src in enumerate(test_cases):
        # 对测试样本编码并移至对应设备
        src_ids = tokenizer.encode(raw_src, max_len=30, add_special=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device) # 添加 Batch 维度
        
        # 贪婪解码
        decoded_ids, attn_weights = greedy_decode(model, src_tensor, tokenizer, max_tgt_len=15)
        pred_str = tokenizer.decode(decoded_ids[0])
        
        print(f"测试样例 {idx + 1} -> 输入日期: {raw_src:<30} | 预测输出 (标准格式): {pred_str}")
        
        # 仅对第一个样例进行注意热力图的展示
        if idx == 0 and attn_weights is not None:
            # 整合多头维度的注意力，变为 [B, L_tgt, L_src]
            avg_attn = get_average_attention(attn_weights)
            # 取第一条数据 (Batch=0) 的二维对齐特征矩阵
            single_attn_matrix = avg_attn[0]
            
            # 展示终端 ASCII 阴影图
            visualize_ascii_attention(raw_src, pred_str, single_attn_matrix)
            
            # 如果安装了 Matplotlib，绘制高清图片
            if HAS_MATPLOTLIB:
                img_path = f"docs/attention_heatmap_{args.model}.png"
                save_matplotlib_heatmap(raw_src, pred_str, single_attn_matrix, img_path)
            else:
                print("【说明】未检测到 matplotlib，已跳过高清图片生成。您可通过终端 ASCII 字符热力图直观查看效果。")


if __name__ == "__main__":
    main()
