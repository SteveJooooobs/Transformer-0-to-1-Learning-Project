# Transformer 学习与对比平台 (Handwritten vs. PyTorch Native)

本项目通过一个直观的**日期格式转换任务**（例如将 `"May 23, 2026"` 翻译并标准化为 `"2026-05-23"`），从零实现了**纯手写 Transformer** 和 **PyTorch 官方封装版 Transformer**，并提供双版本模型在参数结构、训练效率与注意力机制对齐（Attention Alignment Map）上的深度对比。

---

## 🚀 项目亮点

1.  **纯手写白盒实现**：完全不依赖官方 `nn.Transformer`，从零实现位置编码、自写多头注意力、前馈网络和编解码层，清晰展现张量维度流转。
2.  **官方封装与黑科技拦截**：使用 PyTorch 内置的 `nn.Transformer`。利用 **Monkey Patching** 动态代理技术，在不破坏官方底层融合算子（FlashAttention 等）的情况下拦截截获注意力矩阵。
3.  **直观的控制台可视化**：训练仅需在 CPU 上运行 1~2 分钟，即可实现收敛。训练结束后，可在终端以常规 ASCII 阴影字符画直接查看注意力机制如何精准“对齐”输入与输出的日期元素。

---

## 📁 目录结构

本轮开发与重构后的核心文件已全部归档于 `work_01` 主目录下：

```text
Transformer_demo/
├── .gitignore                  # Git 忽略文件（已过滤虚拟环境、模型权重及数据缓存）
├── requirements.txt            # 项目依赖列表文件
├── test_imports.py             # 依赖环境验证脚本
├── README.md                   # 本说明文档
└── work_01/                    # 项目核心工作目录
    ├── dataset.py              # 数据集生成（ procedurally generated ）与字符级分词器
    ├── train.py                # 统一训练、推断测试与 ASCII 可视化主脚本
    ├── models/                 # 模型模块包
    │   ├── __init__.py        # 包初始化与导出
    │   ├── handwritten_transformer.py # 纯手写白盒模型
    │   └── torch_transformer.py # 官方封装模型（含注意力拦截机制）
    ├── docs/                   # 项目主要成果与对比文档
    │   ├── comparison.md      # 手写与官方实现的深度技术比对
    │   └── replication_guide.md # 快速复现运行指南
    └── ai-work/                # 历史实施计划与文档副本
```

---

## 🛠️ 快速开始

想要快速复现和运行本项目，请参考以下简要命令（详细内容请参见 [work_01/docs/replication_guide.md](work_01/docs/replication_guide.md)）：

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 一键安装依赖包
pip install -r requirements.txt

# 3. 运行手写版 Transformer 训练
python work_01/train.py --model handwritten --epochs 5

# 4. 运行官方工具版 Transformer 训练
python work_01/train.py --model torch --epochs 5
```

---

## 📊 技术比对与深度分析

如果您想深入了解手写模型和 PyTorch 官方原生封装（如 GEMM 矩阵合并、快路径 Fast Path、FlashAttention 等）的底层计算差异，请直接阅读文档：
👉 **[手写实现 vs 官方封装对比说明文档](work_01/docs/comparison.md)**

---

## 🤖 AI 标识 (AI-Generated Statement)

> [!NOTE]
> **本仓库中的所有核心代码、算法实现、中英注释、统一测试脚本、对比说明文档以及目录结构重构，均由 AI 编码助手 Antigravity 全量生成并验证。**
