# Transformer 项目复现指南 (Replication Guide)

本指南旨在指导用户如何从零开始，在本地复现和运行 Transformer 的**纯手写实现**与**官方封装工具实现**，并完成训练、推理及注意力机制可视化。

---

## 1. 项目目录结构

在运行代码前，请确保的项目根目录结构如下：
```text
Transformer_demo/
├── .venv/                      # Python 虚拟环境目录
├── requirements.txt            # 项目依赖列表文件
├── test_imports.py             # 依赖库导入验证脚本
└── work_01/                    # 本次重构后的工作主目录
    ├── ai-work/                # 存储实施计划与工作流文件
    │   ├── implementation_plan.md  # 实施计划文档
    │   └── replication_guide.md    # 复现指南文档副本
    ├── docs/                   # 存放项目文档与可视化成果
    │   ├── comparison.md      # 手写与官方对比文档
    │   └── replication_guide.md # 复现指南文档
    ├── models/                 # 模型定义包
    │   ├── __init__.py        # 包初始化与导出
    │   ├── handwritten_transformer.py # 纯手写 Transformer 实现
    │   └── torch_transformer.py # PyTorch 官方封装 Transformer 实现
    ├── dataset.py              # 日期生成与分词器模块
    └── train.py                # 统一训练、评估与可视化脚本
```

---

## 2. 运行环境配置与复现

### 第一步：创建虚拟环境并激活
在项目根目录 `f:/PyProject/Transformer_demo` 下，打开 PowerShell 并执行：
```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1
```

### 第二步：安装核心依赖包
运行以下命令，一键安装研究所需的所有依赖包：
```powershell
# 升级 pip 到最新版本
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 通过 requirements.txt 一键安装所有项目依赖
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
*注：`requirements.txt` 包含了训练与高清热力图绘制所需的所有包（包括 PyTorch, Numpy, Matplotlib 等）。*

### 第三步：验证依赖是否安装成功
运行项目预设的测试导入脚本：
```powershell
.\.venv\Scripts\python.exe test_imports.py
```
若控制台输出“所有核心依赖包导入测试通过！”，即可进入下一步。

---

## 3. 运行模型训练与评估

训练脚本 `train.py` 提供了一系列命令行参数以支持不同场景下的配置。

### 核心可配置参数：
*   `--model`: 选择运行的模型版本，可选 `handwritten`（手写，默认）或 `torch`（官方）。
*   `--epochs`: 训练 Epoch 数（默认 5 轮）。在 CPU 上，5 轮训练通常在 1~2 分钟内即可收敛并达到 98% 以上的整句准确率。
*   `--batch_size`: 批处理大小（默认 128）。
*   `--lr`: 学习率（默认 0.001）。
*   `--num_samples`: 生成的日期样本总数（默认 8000）。

---

### 复现步骤 1：运行纯手写版本模型训练
执行以下命令：
```powershell
.\.venv\Scripts\python.exe train.py --model handwritten --epochs 5
```
**预期输出与结果表现**：
*   控制台将打印模型的层结构及总参数量（约 45.9 万参数）。
*   每个 Epoch 将输出训练 Loss、验证 Loss、Token 级准确率及 Sequence 级（整句完全匹配）准确率。
*   训练结束后，控制台会输出 7 个不同日期输入格式的推理测试样例。
*   针对第一个测试样例 `"May 23, 2026"`，控制台会打印出对应的 **ASCII 注意力热力图**。同时在 `docs/` 目录下生成并保存一张高清的注意力矩阵热力图 `docs/attention_heatmap_handwritten.png`。

---

### 复现步骤 2：运行官方封装版本模型训练
执行以下命令：
```powershell
.\.venv\Scripts\python.exe train.py --model torch --epochs 5
```
**预期输出与结果表现**：
*   控制台打印出官方版本参数规模（与手写版参数量基本一致，约为 45.9 万参数）。
*   进行 5 轮自回归训练，模型能够在第 3~4 个 Epoch 快速收敛至 98% 以上的整句匹配率。
*   训练结束后输出测试集的推理结果，通过 Monkey Patching 拦截技术，脚本会同样在命令行中输出由 `nn.MultiheadAttention` 计算出的 **ASCII 对齐热力图**，并保存高清图像至 `docs/attention_heatmap_torch.png`。

---

## 4. 可视化注意力热力图解读

在运行完训练后，可在 `docs/` 目录下找到生成的 `attention_heatmap_handwritten.png` 与 `attention_heatmap_torch.png`。

在这些热力图中：
*   **纵轴**代表模型的**输出字符**（如从 `<sos>` 开始生成 `"2"` `"0"` `"2"` `"6"` `"-"` `"0"` `"5"` `"-"` `"2"` `"3"`），自上而下自回归产生。
*   **横轴**代表模型的**输入字符**（如原始输入的日期 `"M"` `"a"` `"y"` `" "` `"2"` `"3"` 等）。
*   **对齐关系**：会发现，当模型输出字符 `"2"` `"0"` `"2"` `"6"` 时，横轴上的输入年份 `"2"` `"0"` `"2"` `"6"` 会亮起强关注块（阴影字符画中为 `█` 字符）；当模型输出月份 `"0"` `"5"` 时，输入日期中的 `"M"` `"a"` `"y"` 会亮起强关注。这表明模型已经完美学会了“翻译”和“抽取”日期的关键位置信息！
