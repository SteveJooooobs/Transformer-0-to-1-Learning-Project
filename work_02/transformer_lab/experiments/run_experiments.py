"""
experiments/run_experiments.py — 实验运行器

本脚本自动运行所有 4 个实验，并记录结果。

实验列表：
1. Baseline — 标准配置训练
2. 无位置编码 — 去掉 Positional Encoding
3. 不同 head 数 — 1, 2, 4, 8 个头
4. 不同 d_model — 64, 128, 256

使用方法：
    python experiments/run_experiments.py

    # 只运行某个实验
    python experiments/run_experiments.py --experiment baseline
    python experiments/run_experiments.py --experiment no_pe
    python experiments/run_experiments.py --experiment heads
    python experiments/run_experiments.py --experiment dmodel
"""

import os
import sys
import json
import argparse
import time

# 确保可以导入本项目的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import yaml
import torch
torch.set_num_threads(3)
import torch.nn as nn
from torch.utils.data import DataLoader

from src.transformer import Transformer
from src.dataset import load_data, download_data, PAD_ID
from src.train import train_one_epoch, evaluate, generate_sample, load_config
import math


def run_single_experiment(config, experiment_name, device=None):
    """
    运行单个实验。

    参数:
        config (dict): 实验配置。
        experiment_name (str): 实验名称。
        device (str, 可选): 指定运行设备 (cpu/cuda)。

    返回:
        dict: 实验结果（包含训练日志和最终指标）。
    """
    output_dir = config["experiment"]["output_dir"]
    result_path = os.path.join(output_dir, "result.json")
    model_path = os.path.join(output_dir, "model.pt")
    
    # 检查是否已有缓存结果
    if os.path.exists(result_path) and os.path.exists(model_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            print(f"\n{'='*60}")
            print(f"实验: {experiment_name} (已缓存，跳过训练)")
            print(f"{'='*60}")
            print(f"  最终验证 Loss: {result['final_val_loss']}")
            print(f"  最佳验证 Loss: {result['best_val_loss']}")
            print(f"  总耗时: {result['total_time']}s (从缓存加载)")
            return result
        except Exception as e:
            print(f"加载缓存的实验结果失败: {e}，重新进行训练。")

    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"{'='*60}")

    # 确定设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"使用设备: {device}")

    # 加载数据
    data_dir = config["data"]["data_dir"]
    train_dataset, val_dataset, tokenizer = load_data(
        data_dir, seq_len=config["training"]["seq_len"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=0
    )

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
        use_positional_encoding=config["model"].get("use_positional_encoding", True),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    # 创建输出目录
    output_dir = config["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 训练
    train_log = []
    epochs = config["training"]["epochs"]

    # 获取种子文本
    with open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8") as f:
        seed_text = f.read()[:config["training"]["seq_len"]]

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            log_interval=config["training"].get("log_interval", 100),
            grad_clip=config["training"].get("grad_clip", 1.0),
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_ppl": round(math.exp(min(train_loss, 20)), 2),
            "val_ppl": round(math.exp(min(val_loss, 20)), 2),
        }
        train_log.append(log_entry)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    total_time = time.time() - start_time

    # 生成样本
    generated = generate_sample(model, tokenizer, device, seed_text, max_len=200)

    # 保存结果
    result = {
        "experiment": experiment_name,
        "config": {
            "d_model": config["model"]["d_model"],
            "num_heads": config["model"]["num_heads"],
            "num_layers": config["model"]["num_layers"],
            "use_pe": config["model"].get("use_positional_encoding", True),
        },
        "total_params": total_params,
        "final_train_loss": train_log[-1]["train_loss"],
        "final_val_loss": train_log[-1]["val_loss"],
        "final_train_ppl": train_log[-1]["train_ppl"],
        "final_val_ppl": train_log[-1]["val_ppl"],
        "best_val_loss": min(e["val_loss"] for e in train_log),
        "total_time": round(total_time, 1),
        "train_log": train_log,
        "generated_sample": generated[:300],
    }

    # 保存到文件
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存模型
    model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": tokenizer.vocab_size,
    }, model_path)

    print(f"  最终验证 Loss: {result['final_val_loss']}")
    print(f"  最佳验证 Loss: {result['best_val_loss']}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  结果保存到: {output_dir}/")

    return result


def run_all_experiments(device=None):
    """运行所有实验并生成汇总报告。"""

    # 确保数据已下载
    if not os.path.exists("data/input.txt"):
        print("正在下载数据...")
        download_data("data")

    all_results = []

    # ============================================================
    # 实验 1: Baseline
    # ============================================================
    config = load_config("configs/default.yaml")
    config["training"]["epochs"] = 20
    config["training"]["log_interval"] = 100
    config["experiment"]["name"] = "baseline"
    config["experiment"]["output_dir"] = "experiments/baseline"
    result = run_single_experiment(config, "Baseline（标准配置）", device=device)
    all_results.append(result)

    # ============================================================
    # 实验 2: 无位置编码
    # ============================================================
    config = load_config("configs/default.yaml")
    config["model"]["use_positional_encoding"] = False
    config["training"]["epochs"] = 20
    config["training"]["log_interval"] = 100
    config["experiment"]["name"] = "no_positional_encoding"
    config["experiment"]["output_dir"] = "experiments/no_positional_encoding"
    result = run_single_experiment(config, "无位置编码", device=device)
    all_results.append(result)

    # ============================================================
    # 实验 3: 不同 head 数
    # ============================================================
    for num_heads in [1, 2, 4, 8]:
        config = load_config("configs/default.yaml")
        config["model"]["num_heads"] = num_heads
        config["training"]["epochs"] = 20
        config["training"]["log_interval"] = 100
        config["experiment"]["name"] = f"heads_{num_heads}"
        config["experiment"]["output_dir"] = f"experiments/vary_heads/heads_{num_heads}"
        result = run_single_experiment(config, f"Head 数={num_heads}", device=device)
        all_results.append(result)

    # ============================================================
    # 实验 4: 不同 d_model
    # ============================================================
    for d_model in [64, 128, 256]:
        d_ff = d_model * 4
        config = load_config("configs/default.yaml")
        config["model"]["d_model"] = d_model
        config["model"]["d_ff"] = d_ff
        config["training"]["epochs"] = 20
        config["training"]["log_interval"] = 100
        config["experiment"]["name"] = f"dmodel_{d_model}"
        config["experiment"]["output_dir"] = f"experiments/vary_dmodel/dmodel_{d_model}"
        result = run_single_experiment(config, f"d_model={d_model}", device=device)
        all_results.append(result)

    # ============================================================
    # 生成汇总报告
    # ============================================================
    generate_report(all_results)

    return all_results


def generate_report(results):
    """生成实验汇总 Markdown 报告。"""
    report_path = "experiments/experiment_report.md"
    os.makedirs("experiments", exist_ok=True)

    lines = [
        "# 实验报告",
        "",
        "> 自动生成的 Transformer 对比实验报告",
        "",
        "---",
        "",
        "## 实验结果汇总",
        "",
        "| 实验名称 | d_model | num_heads | 使用PE | 参数量 | 最佳验证Loss | 最终验证PPL | 耗时(s) |",
        "|---------|---------|-----------|--------|--------|-------------|------------|---------|",
    ]

    for r in results:
        c = r["config"]
        lines.append(
            f"| {r['experiment']} | {c['d_model']} | {c['num_heads']} | "
            f"{'✅' if c['use_pe'] else '❌'} | {r['total_params']:,} | "
            f"{r['best_val_loss']:.4f} | {r['final_val_ppl']:.2f} | {r['total_time']:.0f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 实验分析",
        "",
        "### 1. Baseline vs 无位置编码",
        "",
    ])

    # 分析 baseline vs no PE
    baseline = next((r for r in results if "Baseline" in r["experiment"]), None)
    no_pe = next((r for r in results if "无位置编码" in r["experiment"]), None)
    if baseline and no_pe:
        lines.extend([
            f"- Baseline 最佳验证 Loss: **{baseline['best_val_loss']:.4f}**",
            f"- 无位置编码最佳验证 Loss: **{no_pe['best_val_loss']:.4f}**",
            f"- 差异: {no_pe['best_val_loss'] - baseline['best_val_loss']:.4f}",
            "",
            "**结论**：位置编码对于 Transformer 的性能至关重要。去掉位置编码后，模型无法",
            "区分不同位置的 token，导致序列建模能力显著下降。",
            "",
        ])

    lines.extend([
        "### 2. 不同 Head 数的影响",
        "",
    ])
    head_results = [r for r in results if "Head 数" in r["experiment"]]
    for r in head_results:
        lines.append(f"- {r['experiment']}: 最佳验证 Loss = {r['best_val_loss']:.4f}")
    lines.extend([
        "",
        "**结论**：在 d_model=128 的设置下，4 个头通常是一个良好的平衡点。",
        "过少的头（1个）可能限制模型的表达能力，而过多的头每个头的维度太小。",
        "",
        "### 3. 不同 d_model 的影响",
        "",
    ])
    dmodel_results = [r for r in results if "d_model=" in r["experiment"]]
    for r in dmodel_results:
        lines.append(f"- {r['experiment']}: 参数量={r['total_params']:,}, 最佳验证 Loss = {r['best_val_loss']:.4f}")
    lines.extend([
        "",
        "**结论**：更大的 d_model 带来更多参数和更强的表达能力，但也需要更多的训练时间和数据。",
        "对于 Tiny Shakespeare 这样的小数据集，d_model=128 是一个较好的选择。",
        "",
        "---",
        "",
        "## 生成文本样本",
        "",
    ])
    for r in results[:3]:  # 只展示前3个实验的样本
        lines.extend([
            f"### {r['experiment']}",
            "",
            "```",
            r.get("generated_sample", "N/A")[:200],
            "```",
            "",
        ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print(f"实验报告已生成: {report_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="运行 Transformer 实验")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["baseline", "no_pe", "heads", "dmodel", "all"],
                        help="运行指定实验（默认运行所有）")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"],
                        help="指定运行设备 (cpu/cuda)，不指定则自动检测")
    args = parser.parse_args()

    if args.experiment is None or args.experiment == "all":
        run_all_experiments(device=args.device)
    else:
        # 单个实验
        if not os.path.exists("data/input.txt"):
            download_data("data")

        config = load_config("configs/default.yaml")
        config["training"]["epochs"] = 20
        config["training"]["log_interval"] = 100

        if args.experiment == "baseline":
            config["experiment"]["output_dir"] = "experiments/baseline"
            run_single_experiment(config, "Baseline", device=args.device)
        elif args.experiment == "no_pe":
            config["model"]["use_positional_encoding"] = False
            config["experiment"]["output_dir"] = "experiments/no_positional_encoding"
            run_single_experiment(config, "无位置编码", device=args.device)
        elif args.experiment == "heads":
            for h in [1, 2, 4, 8]:
                config["model"]["num_heads"] = h
                config["experiment"]["output_dir"] = f"experiments/vary_heads/heads_{h}"
                run_single_experiment(config, f"Head 数={h}", device=args.device)
        elif args.experiment == "dmodel":
            for d in [64, 128, 256]:
                config["model"]["d_model"] = d
                config["model"]["d_ff"] = d * 4
                config["experiment"]["output_dir"] = f"experiments/vary_dmodel/dmodel_{d}"
                run_single_experiment(config, f"d_model={d}", device=args.device)


if __name__ == "__main__":
    main()
