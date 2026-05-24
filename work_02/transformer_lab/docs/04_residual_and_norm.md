# 残差连接与层归一化

> 本文解释 Transformer 中的两个关键技术：残差连接（Residual Connection）和层归一化（Layer Normalization），它们是深层网络能够稳定训练的基础。

---

## 一、为什么深层网络需要残差连接？

### 深层网络的困境

直觉上，更深的网络应该有更强的表达能力。但在实践中：

```
浅层网络（3 层）：准确率 90%
深层网络（20 层）：准确率 85%  ← 反而更差了！
```

这就是**退化问题（Degradation Problem）**：网络越深，性能反而可能下降。

原因：深层网络在训练时，梯度需要经过很多层回传。每经过一层，梯度都会被乘以一个系数：
- 如果系数 < 1 → 梯度越来越小 → **梯度消失**
- 如果系数 > 1 → 梯度越来越大 → **梯度爆炸**

### 残差连接的解决方案

残差连接（Residual Connection）的核心思想非常简单：

```
普通连接：output = F(x)              ← 直接变换
残差连接：output = x + F(x)          ← 输入 + 变换结果
```

用公式表示：

$$\text{output} = x + \text{SubLayer}(x)$$

其中 $\text{SubLayer}$ 可以是注意力层或前馈网络。

### 为什么有效？

1. **梯度直通路径**：即使 $F(x)$ 的梯度很小，$x$ 的梯度恒为 1，保证了梯度能顺畅地回传
2. **学习残差更容易**：网络只需要学习 $F(x) = \text{期望输出} - x$（残差），而不是直接学习复杂的变换
3. **退化保护**：最差情况下 $F(x) = 0$，残差连接退化为恒等映射，至少不会变差

```
梯度回传路径：

普通连接：  loss → 层N → 层N-1 → ... → 层1     （梯度容易消失）
                ↓      ↓            ↓

残差连接：  loss → 层N → 层N-1 → ... → 层1     （有梯度直通路径）
                 ↘   ↘         ↘
            loss → → → → → → → → → → → 层1     （梯度高速公路）
```

---

## 二、什么是层归一化（Layer Normalization）？

### 归一化的目的

神经网络中，每一层的输出分布可能差异很大。如果上一层的输出值特别大或特别小，下一层就难以稳定地学习。

**归一化的作用**：将数据调整到均值为 0、方差为 1 的标准分布，使训练更稳定。

### LayerNorm vs BatchNorm

两种常见的归一化方式：

| 特性 | Batch Normalization | Layer Normalization |
|------|-------------------|-------------------|
| **归一化维度** | 在 batch 维度上归一化 | 在特征维度上归一化 |
| **依赖关系** | 需要大 batch size | 与 batch size 无关 |
| **适用场景** | CNN（图像） | RNN / Transformer（序列） |
| **推理时** | 需要维护运行统计量 | 不需要额外统计量 |

```
假设输入形状为 [Batch=3, SeqLen=4, Features=5]

BatchNorm: 对每个特征位置，在 batch 维度上计算均值和方差
           ↓ 跨所有样本的同一特征归一化

LayerNorm: 对每个样本的每个位置，在特征维度上计算均值和方差
           ↓ 每个样本独立归一化
```

### Transformer 选择 LayerNorm 的原因

1. **序列长度可变**：不同样本的序列长度不同，BatchNorm 在 batch 内难以对齐
2. **batch size 独立**：LayerNorm 对每个样本独立计算，不受 batch size 影响
3. **更适合自注意力**：自注意力的输出分布因序列不同而差异很大，按样本归一化更合理

### LayerNorm 的公式

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu$ = 特征维度上的均值
- $\sigma^2$ = 特征维度上的方差
- $\epsilon$ = 一个极小值（如 $10^{-5}$），防止除以零
- $\gamma, \beta$ = 可学习的缩放和偏移参数

---

## 三、在 Transformer 中的具体应用

### Post-LN 结构（本项目使用）

这是原始 Transformer 论文中使用的方式：

```python
# 先计算子层，再加残差，最后归一化
attn_output = self_attention(x)
x = LayerNorm(x + Dropout(attn_output))    # 残差 + 归一化

ff_output = feed_forward(x)
x = LayerNorm(x + Dropout(ff_output))       # 残差 + 归一化
```

### Pre-LN 结构（另一种变体）

后来的研究发现，先归一化再计算，训练更稳定：

```python
# 先归一化，再计算子层，最后加残差
x_norm = LayerNorm(x)
attn_output = self_attention(x_norm)
x = x + Dropout(attn_output)               # 只有残差，没有归一化

x_norm = LayerNorm(x)
ff_output = feed_forward(x_norm)
x = x + Dropout(ff_output)                 # 只有残差，没有归一化
```

| 方式 | 优点 | 缺点 |
|------|------|------|
| Post-LN | 论文原版，经典 | 深层时训练可能不稳定 |
| Pre-LN | 训练更稳定 | 输出需要额外的归一化 |

> 💡 本项目使用 Post-LN，因为它更直观且在 2 层的小模型上表现良好。

---

## 四、编码器和解码器中的具体结构

### 编码器层 (EncoderLayer)

```
输入 x
  │
  ├──→ [多头自注意力] → attn_out
  │                        │
  │    (残差连接)  ←────── + ←── Dropout
  │        │
  │    [LayerNorm] → x'
  │
  ├──→ [前馈网络] → ff_out
  │                    │
  │    (残差连接) ←──── + ←── Dropout
  │        │
  │    [LayerNorm] → 输出
```

### 解码器层 (DecoderLayer)

```
输入 x + Memory
  │
  ├──→ [掩码自注意力] → 残差 + LayerNorm → x'
  │
  ├──→ [交叉注意力(x', Memory)] → 残差 + LayerNorm → x''
  │
  ├──→ [前馈网络(x'')] → 残差 + LayerNorm → 输出
```

---

## 五、Dropout 的作用

在残差连接中，我们还使用了 **Dropout** 进行正则化：

```python
x = LayerNorm(x + Dropout(sublayer_output))
```

Dropout 在训练时随机将一部分神经元的输出置为 0：
- **防止过拟合**：迫使网络学习更鲁棒的特征
- **增加多样性**：不同的训练步使用不同的网络子结构

> 💡 Dropout 只在训练时生效，推理时不使用。本项目默认 dropout=0.1。

---

## 六、小结

| 技术 | 解决的问题 | 关键公式 |
|------|-----------|---------|
| **残差连接** | 梯度消失、退化问题 | $\text{output} = x + F(x)$ |
| **LayerNorm** | 训练不稳定、内部协变量偏移 | $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$ |
| **Dropout** | 过拟合 | 随机置零部分输出 |

这三个技术组合在一起，使 Transformer 能够稳定地训练更深的网络，获得更强的性能。
