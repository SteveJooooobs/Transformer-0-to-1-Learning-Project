# From-Scratch Transformer Lab

> 从零实现 Transformer 教学实验室 —— 一个面向初学者的完整教学工程项目

---

## 🎯 项目目标

让一个完全没有 Transformer 实现经验的人，仅依靠项目内文档，就能从零完成环境搭建、代码运行、原理理解和实验验证。

---

## 📁 项目结构

```
transformer_lab/
├── src/                          # 源代码
│   ├── __init__.py               # 包初始化
│   ├── embedding.py              # 词嵌入层
│   ├── positional_encoding.py    # 位置编码
│   ├── attention.py              # 缩放点积注意力
│   ├── multi_head_attention.py   # 多头注意力
│   ├── encoder.py                # 编码器（含前馈网络）
│   ├── decoder.py                # 解码器
│   ├── transformer.py            # 完整 Transformer 模型
│   ├── dataset.py                # 数据集加载与字符分词器
│   ├── train.py                  # 训练脚本
│   └── inference.py              # 推理脚本
├── configs/                      # 配置文件
│   ├── default.yaml              # 默认配置
│   ├── experiment_no_pe.yaml     # 去掉位置编码实验
│   ├── experiment_heads.yaml     # 修改 head 数实验
│   └── experiment_dmodel.yaml    # 修改 d_model 实验
├── tests/                        # 测试
│   ├── test_shapes.py            # 形状测试
│   ├── test_forward.py           # 前向传播测试
│   ├── test_attention.py         # 注意力正确性测试
│   ├── test_training.py          # 训练烟雾测试
│   └── test_inference.py         # 推理测试
├── docs/                         # 教学文档
│   ├── 01_transformer_overview.md
│   ├── 02_attention.md
│   ├── 03_multi_head.md
│   ├── 04_residual_and_norm.md
│   └── 05_training.md
├── data/                         # 数据目录
├── experiments/                  # 实验结果
├── notebooks/                    # Jupyter Notebooks
├── guide.md                      # 从零实践指南 ⭐
├── README.md                     # 本文件
└── requirements.txt              # 依赖列表
```

---

## 🚀 快速开始

```bash
# 1. 进入项目目录
cd work_02/transformer_lab

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载数据
python -m src.train --download

# 5. 运行测试
python -m pytest tests/ -v

# 6. 开始训练
python -m src.train

# 7. 生成文本
python -m src.inference --model experiments/baseline/model_best.pt --interactive
```

---

## 🧩 模型架构

```
源序列 → [词嵌入 + 位置编码] → [编码器×2] → Memory
                                                ↓
目标序列 → [词嵌入 + 位置编码] → [解码器×2] → [线性层] → 输出
```

默认配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 128 | 模型隐藏维度 |
| num_heads | 4 | 注意力头数 |
| d_ff | 512 | 前馈网络维度 |
| num_layers | 2 | 编码器/解码器层数 |
| seq_len | 64 | 序列长度 |
| vocab_size | ~68 | 字符级词表 |

---

## 📖 文档

按以下顺序阅读：

1. [Transformer 概述](docs/01_transformer_overview.md)
2. [注意力机制详解](docs/02_attention.md)
3. [多头注意力](docs/03_multi_head.md)
4. [残差连接与层归一化](docs/04_residual_and_norm.md)
5. [训练过程详解](docs/05_training.md)
6. [从零实践指南](guide.md) ⭐

---

## 🧪 实验

| 实验 | 目的 | 配置文件 |
|------|------|---------|
| Baseline | 标准训练基线 | `configs/default.yaml` |
| 去掉位置编码 | 验证位置信息的重要性 | `configs/experiment_no_pe.yaml` |
| 修改 Head 数 | 比较 1/2/4/8 个 head | `configs/experiment_heads.yaml` |
| 修改 d_model | 比较 64/128/256 维度 | `configs/experiment_dmodel.yaml` |

---

## 💻 硬件要求

- **最低**：CPU + 4GB 内存
- **推荐**：NVIDIA GPU + 8GB 显存
- 默认配置下 CPU 训练约 30-60 分钟（30 epoch）
- 支持 `--device cpu` / `--device cuda` 切换

---

## 📝 许可

本项目仅供教学使用。
