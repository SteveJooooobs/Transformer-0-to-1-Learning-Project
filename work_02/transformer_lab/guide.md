# From-Scratch Transformer Lab — 从零开始实践指南

> 本指南面向完全没有 Transformer 实现经验的初学者。跟着本指南一步步操作，你将从零完成环境搭建、代码运行、原理理解和实验验证。

---

## 📋 目录

1. [环境搭建](#1-环境搭建)
2. [数据准备](#2-数据准备)
3. [运行训练](#3-运行训练)
4. [运行推理](#4-运行推理)
5. [运行测试](#5-运行测试)
6. [运行实验](#6-运行实验)
7. [调试方法](#7-调试方法)
8. [常见错误](#8-常见错误)

---

## 1. 环境搭建

### 第一步：确认 Python 版本

打开终端（命令行），输入：

```bash
python --version
```

你应该看到 `Python 3.8` 或更高版本。如果没有 Python，请先到 [python.org](https://www.python.org/) 下载安装。

### 第二步：创建虚拟环境

虚拟环境是一个独立的 Python 环境，可以避免不同项目之间的包冲突。

```bash
# 进入项目目录
cd work_02/transformer_lab

# 创建虚拟环境（名为 .venv）
python -m venv .venv
```

**为什么要用虚拟环境？**
- 隔离项目依赖，不污染系统 Python
- 不同项目可以使用不同版本的库
- 方便重现环境

### 第三步：激活虚拟环境

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

激活成功后，终端提示符前会出现 `(.venv)` 标志。

### 第四步：安装依赖

```bash
pip install -r requirements.txt
```

这会安装：
- `torch`：PyTorch 深度学习框架
- `pyyaml`：YAML 配置文件解析
- `pytest`：测试框架
- `numpy`：数值计算
- `matplotlib`：可视化
- `tqdm`：进度条

安装完成后验证：

```bash
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

> 💡 **特别说明：CPU 版本 vs GPU 版本 PyTorch**
>
> 默认运行 `pip install -r requirements.txt` 安装的 PyTorch 可能是 CPU 版本。如果你拥有 NVIDIA 显卡并希望使用 GPU 进行极速训练（如 RTX 4060），可以根据你的 CUDA 版本手动安装 GPU 版 PyTorch（例如支持 CUDA 12.1 的版本）：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
> ```
> 验证输出中 `CUDA 可用: True` 即表示成功切换为 GPU 运行环境。

---

## 2. 数据准备

### 关于 Tiny Shakespeare

我们使用 **Tiny Shakespeare** 数据集——约 1MB 的莎士比亚戏剧全文。

**为什么选它？**
- 数据量适中，CPU 也能训练
- 文本有明显的风格和格式
- 生成结果可以直接阅读来判断模型质量
- NLP 教学中最经典的数据集

### 下载数据

```bash
python -m src.train --download
```

这会自动下载数据到 `data/input.txt`。

**如果下载失败**（网络问题），可以手动下载：
1. 在浏览器打开：https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
2. 保存文件到 `data/input.txt`

### 验证数据

```bash
# 查看文件大小和前几行
python -c "
with open('data/input.txt', 'r') as f:
    text = f.read()
print(f'文件大小: {len(text):,} 字符')
print(f'唯一字符数: {len(set(text))}')
print(f'前 200 个字符:')
print(text[:200])
"
```

你应该看到大约 1,115,394 个字符，约 65 个不同字符。

### 数据格式

数据是纯文本文件，内容是莎士比亚的戏剧：

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You know, Caius Marcius is chief enemy to the people.
```

我们的任务是：给定一段文本（64 个字符），预测接下来的 64 个字符。

---

## 3. 运行训练

### 使用默认配置训练

```bash
python -m src.train
```

### 使用自定义配置

```bash
python -m src.train --config configs/default.yaml
```

### 指定设备

```bash
# 强制使用 CPU
python -m src.train --device cpu

# 使用 GPU（如果可用）
python -m src.train --device cuda
```

### 训练过程中会看到什么？

```
==================================================
加载数据
==================================================
数据集统计:
  总字符数: 1,115,394
  唯一字符数: 65
  词表大小: 68

==================================================
构建模型
==================================================
模型参数量: 1,234,567

==================================================
开始训练
==================================================
  Epoch 1 | Batch 50/245 | Loss: 3.2145 | Avg Loss: 3.4562 | PPL: 31.69
  ...

==================================================
Epoch 1/30 完成 | 耗时: 45.2s
  训练 Loss: 2.8901 | 训练 PPL: 18.00
  验证 Loss: 2.7543 | 验证 PPL: 15.71
```

### 如何判断训练正常？

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| 初始 Loss | ~4.2（≈ log(68)，68 是词表大小） | 如果远高于或远低于这个值 |
| Loss 趋势 | 持续下降 | 不下降或上升 |
| 最终 Loss | 1.5 ~ 2.5 | Loss 变成 NaN |
| 训练时间 | CPU: 每 epoch 30~120 秒 | 时间异常长 |
| 生成文本 | 逐渐出现英文词汇 | 全是乱码 |

### 训练完成后会生成

```
experiments/baseline/
├── model.pt            # 最终模型
├── model_best.pt       # 最佳模型（验证 Loss 最低）
├── train_log.json      # 训练日志
└── config.yaml         # 训练配置
```

---

## 4. 运行推理

### 使用训练好的模型生成文本

```bash
# 使用默认种子文本
python -m src.inference --model experiments/baseline/model_best.pt

# 指定种子文本
python -m src.inference --model experiments/baseline/model_best.pt --seed "ROMEO:"

# 调整生成参数
python -m src.inference --model experiments/baseline/model_best.pt --length 500 --temperature 0.8
```

### 交互模式

```bash
python -m src.inference --model experiments/baseline/model_best.pt --interactive
```

在交互模式下，你可以：
- 输入任意文本作为种子
- 输入 `temp=0.5` 调整温度
- 输入 `len=200` 调整生成长度
- 输入 `quit` 退出

### 温度参数说明

| 温度 | 效果 |
|------|------|
| 0.1 ~ 0.5 | 保守，倾向于最常见的字符，重复性高 |
| 0.7 ~ 0.9 | 平衡，推荐的默认范围 |
| 1.0 | 按原始概率采样 |
| 1.2+ | 更随机，更有创意，但可能出现乱码 |

---

## 5. 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 只运行某一类测试
python -m pytest tests/test_shapes.py -v        # 形状测试
python -m pytest tests/test_attention.py -v      # 注意力正确性
python -m pytest tests/test_training.py -v       # 训练烟雾测试
```

所有 52 个测试都应该通过（显示绿色的 PASSED）。

---

## 6. 运行实验

项目包含 4 个实验，用于验证 Transformer 不同组件的作用：

```bash
# 运行所有实验
python experiments/run_experiments.py

# 或手动运行单个实验（参见 experiments/ 目录下的脚本）
```

实验内容：
1. **Baseline**：标准配置训练
2. **去掉位置编码**：观察位置信息的重要性
3. **修改 head 数量**：1 → 2 → 4 → 8
4. **修改 d_model**：64 → 128 → 256

每个实验会记录 Loss 变化和分析结果。

---

## 7. 调试方法

### 方法 1：检查 tensor 形状

这是最重要的调试技巧！99% 的错误都是形状不匹配。

```python
print(x.shape)  # 查看张量形状
# 例如: torch.Size([2, 64, 128])
# 含义: [batch_size=2, seq_len=64, d_model=128]
```

### 方法 2：逐模块测试

每个模块文件都可以单独运行自测：

```bash
python -m src.embedding           # 测试嵌入层
python -m src.positional_encoding # 测试位置编码
python -m src.attention           # 测试注意力
python -m src.multi_head_attention # 测试多头注意力
python -m src.encoder             # 测试编码器
python -m src.decoder             # 测试解码器
python -m src.transformer         # 测试完整模型
```

### 方法 3：打印中间值

```python
# 在 forward 方法中添加打印
def forward(self, x):
    print(f"输入: {x.shape}, 值范围: [{x.min():.3f}, {x.max():.3f}]")
    # ... 计算 ...
    print(f"输出: {output.shape}, 值范围: [{output.min():.3f}, {output.max():.3f}]")
    return output
```

### 方法 4：检查梯度

```python
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

---

## 8. 常见错误

### ❌ ModuleNotFoundError: No module named 'src'

**原因**：当前目录不在 `transformer_lab/` 下，Python 找不到 `src` 模块。

**解决**：
```bash
# 确保你在正确的目录
cd work_02/transformer_lab

# 使用 -m 方式运行
python -m src.train
```

### ❌ CUDA out of memory

**原因**：模型或 batch 太大，GPU 内存不足。

**解决**：
1. 减小 `batch_size`：在 `configs/default.yaml` 中改小
2. 减小 `d_model`：降低模型大小
3. 减小 `seq_len`：缩短序列长度
4. 使用 CPU：`python -m src.train --device cpu`

### ❌ RuntimeError: mat1 and mat2 shapes cannot be multiplied

**原因**：tensor shape 不匹配，通常是配置参数不一致。

**解决**：
1. 确保 `d_model` 能被 `num_heads` 整除
2. 检查 `src` 和 `tgt` 的序列长度是否一致
3. 使用 `print(x.shape)` 追踪形状变化

### ❌ Loss 不下降

**可能原因及解决**：

| 原因 | 解决方法 |
|------|---------|
| 学习率太大 | 降低到 0.0001 |
| 学习率太小 | 提高到 0.01 |
| 模型太小 | 增大 d_model 或 num_layers |
| 数据问题 | 检查数据是否正确加载 |
| 梯度爆炸 | 启用梯度裁剪 |

### ❌ Loss 变成 NaN

**原因**：数值溢出，通常由学习率过大或梯度爆炸引起。

**解决**：
1. 降低学习率
2. 启用梯度裁剪（默认已启用，max_norm=1.0）
3. 检查数据中是否有异常值

### ❌ FileNotFoundError: data/input.txt

**原因**：数据文件未下载。

**解决**：
```bash
python -m src.train --download
```

---

## 📖 推荐学习路径

1. **阅读文档**：按顺序阅读 `docs/` 目录下的 5 个文档
2. **看代码**：按 embedding → attention → multi_head_attention → encoder → decoder → transformer 的顺序
3. **跑测试**：运行 `pytest tests/ -v`，理解每个测试在验证什么
4. **训练模型**：运行 `python -m src.train`，观察训练过程
5. **生成文本**：运行推理脚本，观察生成效果
6. **做实验**：运行实验，理解每个组件的作用

祝你学习愉快！🎉
