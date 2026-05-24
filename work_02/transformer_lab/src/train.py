"""
train.py — 训练脚本

本脚本负责完整的 Transformer 模型训练流程：
1. 加载配置文件（YAML 格式）
2. 下载/加载数据集
3. 构建模型
4. 执行训练循环
5. 保存模型和训练日志

使用方法：
    # 下载数据集
    python -m src.train --download

    # 使用默认配置训练
    python -m src.train

    # 使用自定义配置训练
    python -m src.train --config configs/default.yaml

    # 指定设备
    python -m src.train --device cpu
    python -m src.train --device cuda

运行后会在 experiments/ 目录下生成：
    - model.pt：模型权重
    - train_log.json：训练日志（loss 记录）
    - config.yaml：本次训练使用的配置
"""

import os
import sys
import json
import time
import math
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 确保可以导入本项目的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.dataset import load_data, download_data, PAD_ID


def get_default_config():
    """
    获取默认训练配置。

    返回:
        dict: 包含模型和训练超参数的字典。
    """
    return {
        "model": {
            "d_model": 128,
            "num_heads": 4,
            "d_ff": 512,
            "num_layers": 2,
            "max_len": 256,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": 30,
            "seq_len": 64,
            "train_ratio": 0.9,
            "grad_clip": 1.0,
            "log_interval": 50,
            "save_interval": 5,
        },
        "data": {
            "data_dir": "data",
        },
        "experiment": {
            "name": "baseline",
            "output_dir": "experiments/baseline",
        }
    }


def load_config(config_path=None):
    """
    加载训练配置。如果没有指定配置文件，使用默认配置。

    参数:
        config_path (str, 可选): YAML 配置文件路径。

    返回:
        dict: 训练配置字典。
    """
    config = get_default_config()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
        # 递归合并用户配置
        _deep_update(config, user_config)
        print(f"已加载配置文件: {config_path}")
    else:
        print("使用默认配置")

    return config


def _deep_update(base, update):
    """递归地更新嵌套字典。"""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch,
                    log_interval=50, grad_clip=1.0):
    """
    执行一个 epoch 的训练。

    参数:
        model: Transformer 模型。
        dataloader: 训练数据加载器。
        optimizer: 优化器。
        criterion: 损失函数。
        device: 计算设备。
        epoch (int): 当前 epoch 编号。
        log_interval (int): 每隔多少个 batch 打印一次日志。
        grad_clip (float): 梯度裁剪阈值。

    返回:
        float: 本 epoch 的平均训练损失。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        # 将数据移到指定设备
        # src: [batch_size, seq_len]
        # tgt: [batch_size, seq_len]
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher Forcing 的输入输出构造：
        # 解码器输入 = 目标序列去掉最后一个 token（右移一位）
        # 训练标签 = 目标序列去掉第一个 token
        # 这样解码器在位置 i 时预测的是位置 i+1 的 token
        tgt_input = tgt[:, :-1]    # [B, seq_len - 1]
        tgt_output = tgt[:, 1:]    # [B, seq_len - 1]

        # 前向传播
        # logits: [B, seq_len - 1, vocab_size]
        logits, _ = model(src, tgt_input)

        # 计算损失
        # 需要将 logits 展平为 [B * (seq_len-1), vocab_size]
        # 将 tgt_output 展平为 [B * (seq_len-1)]
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # 参数更新
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 打印训练日志
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            ppl = math.exp(min(avg_loss, 20))  # 困惑度，防止溢出
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | PPL: {ppl:.2f}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    在验证集上评估模型。

    参数:
        model: Transformer 模型。
        dataloader: 验证数据加载器。
        criterion: 损失函数。
        device: 计算设备。

    返回:
        float: 验证集平均损失。
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits, _ = model(src, tgt_input)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def generate_sample(model, tokenizer, device, seed_text, max_len=100):
    """
    使用训练好的模型生成文本样本（用于训练过程中的效果观察）。

    参数:
        model: Transformer 模型。
        tokenizer: 字符分词器。
        device: 计算设备。
        seed_text (str): 种子文本（源序列）。
        max_len (int): 最大生成长度。

    返回:
        str: 生成的文本。
    """
    model.eval()
    with torch.no_grad():
        # 编码源序列
        src_ids = tokenizer.encode(seed_text)
        src = torch.tensor([src_ids], dtype=torch.long).to(device)

        # 初始化目标序列，以源序列的最后一个字符开始
        tgt_ids = [src_ids[-1]] if src_ids else [PAD_ID]

        for _ in range(max_len):
            tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            logits, _ = model(src, tgt)

            # 取最后一个位置的预测
            # logits[:, -1, :] -> [1, vocab_size]
            next_token = logits[:, -1, :].argmax(dim=-1).item()

            tgt_ids.append(next_token)

            # 如果生成了 EOS，停止
            if next_token == 2:  # EOS_ID
                break

        # 解码生成的文本
        generated = tokenizer.decode(tgt_ids)
        return generated


def main():
    """主训练函数。"""
    # ---- 解析命令行参数 ----
    parser = argparse.ArgumentParser(description="训练 Transformer 模型")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (cpu/cuda)")
    parser.add_argument("--download", action="store_true",
                        help="下载 Tiny Shakespeare 数据集")
    args = parser.parse_args()

    # ---- 加载配置 ----
    config = load_config(args.config)

    # 确定项目根目录（transformer_lab/）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # ---- 处理数据下载 ----
    data_dir = config["data"]["data_dir"]
    if args.download:
        download_data(data_dir)
        print("数据下载完成！")
        return

    # ---- 确定设备 ----
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n使用设备: {device}")

    # ---- 加载数据 ----
    print("\n" + "=" * 50)
    print("加载数据")
    print("=" * 50)

    train_dataset, val_dataset, tokenizer = load_data(
        data_dir,
        seq_len=config["training"]["seq_len"],
        train_ratio=config["training"]["train_ratio"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ---- 构建模型 ----
    print("\n" + "=" * 50)
    print("构建模型")
    print("=" * 50)

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        num_layers=config["model"]["num_layers"],
        max_len=config["model"]["max_len"],
        dropout=config["model"]["dropout"],
        pad_id=PAD_ID,
        use_positional_encoding=config["model"].get("use_positional_encoding", True),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"模型配置: d_model={config['model']['d_model']}, "
          f"num_heads={config['model']['num_heads']}, "
          f"num_layers={config['model']['num_layers']}, "
          f"d_ff={config['model']['d_ff']}")

    # ---- 损失函数和优化器 ----
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    # ---- 创建输出目录 ----
    output_dir = config["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # ---- 训练循环 ----
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50)

    train_log = []
    best_val_loss = float("inf")
    epochs = config["training"]["epochs"]

    # 获取一段种子文本用于生成样本
    with open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8") as f:
        full_text = f.read()
    seed_text = full_text[:config["training"]["seq_len"]]

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            log_interval=config["training"]["log_interval"],
            grad_clip=config["training"]["grad_clip"],
        )

        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        print(
            f"\n{'='*50}\n"
            f"Epoch {epoch}/{epochs} 完成 | 耗时: {elapsed:.1f}s\n"
            f"  训练 Loss: {train_loss:.4f} | 训练 PPL: {train_ppl:.2f}\n"
            f"  验证 Loss: {val_loss:.4f} | 验证 PPL: {val_ppl:.2f}"
        )

        # 记录训练日志
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_ppl": round(val_ppl, 2),
            "time": round(elapsed, 1),
        }
        train_log.append(log_entry)

        # 生成文本样本
        if epoch % config["training"]["save_interval"] == 0 or epoch == epochs:
            generated = generate_sample(model, tokenizer, device, seed_text, max_len=200)
            print(f"\n  生成样本:\n  {'─' * 40}")
            for line in generated[:200].split("\n")[:5]:
                print(f"  {line}")
            print(f"  {'─' * 40}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, "model_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config,
                "vocab_size": tokenizer.vocab_size,
            }, model_path)
            print(f"  💾 保存最佳模型到 {model_path} (val_loss: {val_loss:.4f})")

        # 定期保存模型
        if epoch % config["training"]["save_interval"] == 0:
            model_path = os.path.join(output_dir, f"model_epoch{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config,
                "vocab_size": tokenizer.vocab_size,
            }, model_path)

    # ---- 保存最终模型和日志 ----
    final_model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "val_loss": val_loss,
        "config": config,
        "vocab_size": tokenizer.vocab_size,
    }, final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")

    # 保存训练日志
    log_path = os.path.join(output_dir, "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)
    print(f"训练日志已保存到: {log_path}")

    # ---- 训练总结 ----
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    print(f"  最佳验证 Loss: {best_val_loss:.4f}")
    print(f"  最佳验证 PPL: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"  最终训练 Loss: {train_log[-1]['train_loss']}")
    print(f"  模型保存位置: {output_dir}/")


if __name__ == "__main__":
    main()
