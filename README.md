# Transformer 学习与对比平台 (Handwritten vs. PyTorch Native & From-Scratch Lab)

本项目是一个功能完备、文档详实的 **Transformer 学习与对比实验平台**。项目分为两个核心工作区：

1. **`work_01` (Date Standardization Lab)**: 基于日期格式转换任务（如将 `"May 23, 2026"` 转换为 `"2026-05-23"`），深度对比**纯手写白盒 Transformer** 与 **PyTorch 官方原生封装 Transformer** 在参数结构、训练效率和注意力对齐矩阵（Attention Alignment Map）上的差异。
2. **`work_02` (From-Scratch Transformer Lab)**: 基于 **Tiny Shakespeare** 字符级语言建模任务，从零构建经典的 **Encoder-Decoder Transformer** 教学实验室，配有完备的自动化单元测试、可视化分析 Notebook 以及多维度对比实验（包括无位置编码、不同注意力头数、不同隐藏维度）。

---

## 🚀 项目亮点

*   **双版本对比 (`work_01`)**: 纯手写实现展示清晰的张量维度流转；官方封装版利用 Monkey Patching 动态代理技术，在不破坏底层优化（FlashAttention 等）的情况下拦截并可视化注意力权重。
*   **白盒教学实践 (`work_02`)**: 完全独立的模块化设计（embedding, positional encoding, multi-head attention 等），无任何伪代码，极其适合初学者逐模块阅读与调试。
*   **多维度对比实验 (`work_02/experiments/`)**: 包含 4 大方向、9 个对比实验，定量揭示位置编码、注意力头数、隐藏层维度对生成模型效果及复杂度的直接影响。
*   **一键 CPU/GPU 切换**: 核心运行脚本均已完美支持 `--device cpu` 与 `--device cuda` 一键切换，适配各种计算资源环境。

---

## 📁 目录结构

```text
Transformer_demo/
├── .gitignore                  # Git 忽略文件（过滤虚拟环境、模型权重及缓存）
├── requirements.txt            # 项目依赖列表文件
├── test_imports.py             # 依赖环境验证脚本
├── README.md                   # 本说明文档 (Root)
│
├── work_01/                    # 工作区 01：日期格式标准化与模型对比
│   ├── dataset.py              # 数据集生成与字符分词器
│   ├── train.py                # 统一训练、推断测试与 ASCII 字符画对齐可视化主脚本
│   ├── models/                 # 模型模块包
│   │   ├── handwritten_transformer.py # 纯手写白盒模型
│   │   └── torch_transformer.py       # 官方封装模型（含注意力拦截机制）
│   └── docs/                   # 项目主要成果与对比文档
│       ├── comparison.md      # 手写与官方实现的深度技术比对说明
│       └── replication_guide.md # 快速复现运行指南
│
└── work_02/                    # 工作区 02：从零实现 Transformer 教学实验室
    └── transformer_lab/
        ├── src/                # 模块化白盒源代码
        │   ├── embedding.py, positional_encoding.py, attention.py, ...
        │   ├── transformer.py  # 完整 Transformer 模型结构 (包含 PE 显式控制)
        │   ├── train.py        # 字符级 Seq2Seq 语言模型训练脚本
        │   └── inference.py    # 交互式文本生成推理脚本
        ├── configs/            # 4 大实验方向的 YAML 配置文件
        ├── tests/              # 涵盖模型各层的 52 个 pytest 自动化单元测试
        ├── docs/               # 深入浅出的 5 篇原理剖析文档
        ├── notebooks/          # 可视化注意力热图及训练曲线的 Jupyter Notebooks
        ├── experiments/        # 实验运行器及自动生成的对比实验报告
        ├── guide.md            # 面向新人的快速搭建与调试指南 ⭐
        └── README.md           # work_02 子项目说明文档
```

---

## 🛠️ 快速开始

### 运行工作区 01 (日期格式标准化模型对比)

详细复现步骤请参阅 [work_01/docs/replication_guide.md](work_01/docs/replication_guide.md)：

```powershell
# 1. 激活虚拟环境并安装依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

# 2. 运行手写版 Transformer 训练与可视化
python work_01/train.py --model handwritten --epochs 5

# 3. 运行官方封装版 Transformer 训练与可视化
python work_01/train.py --model torch --epochs 5
```

### 运行工作区 02 (From-Scratch 教学实验室)

详细指引请参阅 [work_02/transformer_lab/guide.md](work_02/transformer_lab/guide.md)：

```powershell
# 1. 进入子目录
cd work_02/transformer_lab

# 2. 下载 Tiny Shakespeare 数据集
python -m src.train --download

# 3. 运行 52 个自动化单元测试，确保模型逻辑完全正确
python -m pytest tests/ -v

# 4. 启动默认基线模型训练
python -m src.train --device cuda   # 优先使用 GPU 训练，也可指定 cpu

# 5. 运行对比实验（一键在 GPU 上跑完所有 9 个配置并自动生成报告）
python experiments/run_experiments.py --device cuda

# 6. 使用训练好的模型，启动交互式文本生成生成器
python -m src.inference --model experiments/baseline/model_best.pt --interactive
```

---

## 📊 成果报告与实验结论

关于工作区 02 运行 9 项对比实验的最终数据和结论，可直接参阅：
👉 **[From-Scratch Lab 自动生成对比实验报告](work_02/transformer_lab/experiments/experiment_report.md)**

---

## 🤖 AI 标识 (AI-Generated Statement)

> [!NOTE]
> **本仓库中的所有核心代码、算法实现、单元测试、对比实验框架、可视化脚本、中英文注释、系列说明文档以及目录结构，均由 AI 编码助手 Antigravity 全量生成并多轮验证。[使用模型:Claude Opus 4.6(Thinking) && Gemini 3.5 Flush(High)]。**


