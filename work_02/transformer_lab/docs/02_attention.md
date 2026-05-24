# 注意力机制详解

> 本文深入解析 Transformer 的核心运算——缩放点积注意力（Scaled Dot-Product Attention），帮助你理解 Q、K、V 的含义以及公式中每个部分的作用。

---

## 一、什么是注意力机制？

在日常生活中，当你阅读一段文字时，你的注意力不会均匀分配到每个字上。比如阅读：

> "**猫**坐在**垫子**上"

当你理解「坐」这个动作时，你的注意力会更多地放在「猫」（谁在坐）和「垫子」（坐在哪里）上。

**注意力机制做的就是这件事**：让模型在处理某个位置时，能够自动学习「应该关注序列中的哪些位置」。

---

## 二、Query、Key、Value —— 图书馆查书的比喻

理解 Q、K、V 的最佳方式是想象你在图书馆找书：

| 概念 | 图书馆比喻 | 在 Transformer 中 |
|------|-----------|-------------------|
| **Query (Q)** 查询 | 你心中想找的主题："我想找关于机器学习的书" | 当前位置想要什么信息 |
| **Key (K)** 键 | 每本书封面上的标签/关键词 | 每个位置能提供什么信息 |
| **Value (V)** 值 | 书的实际内容 | 每个位置实际携带的信息 |
| **注意力权重** | 你的查询与每本书标签的匹配程度 | Q 和 K 的相似度（点积） |
| **输出** | 根据匹配程度，综合所有相关书的内容 | 用权重对所有 V 加权求和 |

> 💡 **关键直觉**：Query 决定了「我在找什么」，Key 决定了「我能提供什么」，两者的匹配程度决定了最终取多少 Value。

---

## 三、缩放点积注意力公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 逐步拆解

#### 第 1 步：计算相似度 $QK^T$

```python
# Q: [B, H, L_q, d_k]  —— 每个位置的查询向量
# K: [B, H, L_k, d_k]  —— 每个位置的键向量
scores = torch.matmul(Q, K.transpose(-2, -1))
# scores: [B, H, L_q, L_k]  —— 每对 (q, k) 之间的相似度
```

矩阵乘法 $QK^T$ 计算的是每个 Query 向量与每个 Key 向量之间的**点积**（dot product）。点积越大，说明两个向量越「相似」，即这两个位置越相关。

#### 第 2 步：缩放 $\div \sqrt{d_k}$

```python
scores = scores / math.sqrt(d_k)
```

**为什么要除以 $\sqrt{d_k}$？**

当 $d_k$ 很大时，Q 和 K 的点积结果的**方差**也会很大。具体来说：
- 假设 Q 和 K 的每个元素都是均值 0、方差 1 的随机变量
- 它们的点积是 $d_k$ 个乘积之和，方差为 $d_k$
- 当 $d_k = 64$ 时，点积的标准差约为 8

如果不缩放，softmax 的输入值会过大，导致：
- softmax 输出趋向于 one-hot（几乎全为 0，只有一个接近 1）
- 梯度变得极小，训练难以进行

除以 $\sqrt{d_k}$ 将方差恢复为 1，让 softmax 保持在合理范围内。

#### 第 3 步：Softmax 归一化

```python
attn_weights = F.softmax(scores, dim=-1)
# attn_weights: [B, H, L_q, L_k]  —— 每行之和为 1
```

Softmax 将原始分数转换为**概率分布**（所有值非负，且每行之和为 1）。

```
原始分数:  [2.0, 1.0, 0.1]
Softmax:   [0.66, 0.24, 0.10]  ← 和为 1
```

这意味着：当前 Query 位置会把 66% 的注意力放在第一个位置，24% 放在第二个，10% 放在第三个。

#### 第 4 步：加权求和

```python
output = torch.matmul(attn_weights, V)
# output: [B, H, L_q, d_k]
```

用注意力权重对 Value 进行加权组合：

$$\text{output}_i = \sum_j \text{weight}_{ij} \cdot V_j$$

每个位置的输出 = 所有位置的 Value 向量的加权平均。

---

## 四、掩码（Mask）的作用

掩码用于强制模型忽略某些位置。有两种常见的掩码：

### 1. Padding Mask（填充掩码）

当一个 batch 中的序列长度不一时，短序列会用 `<pad>` 填充到统一长度。
Padding 位置不包含有意义的信息，必须被忽略。

```
原始序列: ["I", "love", "AI", "<pad>", "<pad>"]
掩码:     [ ✓,    ✓,    ✓,    ✗,       ✗    ]
```

### 2. Causal Mask（因果掩码 / 前瞻掩码）

在解码器中，预测第 $t$ 个 token 时不能看到 $t$ 之后的 token。

```
       位置0  位置1  位置2  位置3
位置0 [  ✓     ✗     ✗     ✗  ]   ← 只能看自己
位置1 [  ✓     ✓     ✗     ✗  ]   ← 只能看 0 和自己
位置2 [  ✓     ✓     ✓     ✗  ]   ← 只能看 0、1 和自己
位置3 [  ✓     ✓     ✓     ✓  ]   ← 可以看所有位置
```

这就是一个**下三角矩阵**（lower triangular matrix）。

### 掩码的实现

```python
# 将被掩码位置的分数设为极大负数
scores = scores.masked_fill(mask == False, -1e9)
# 经过 softmax 后，这些位置的权重趋近于 0
```

---

## 五、代码与公式的对应

以下是 `src/attention.py` 中核心函数与公式的一一对应：

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)

    # 公式: QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 公式: QK^T / √d_k
    scores = scores / math.sqrt(d_k)

    # 掩码处理
    if mask is not None:
        scores = scores.masked_fill(mask == False, -1e9)

    # 公式: softmax(QK^T / √d_k)
    attn_weights = F.softmax(scores, dim=-1)

    # 公式: softmax(QK^T / √d_k) · V
    output = torch.matmul(attn_weights, V)

    return output, attn_weights
```

> 💡 **建议**：对照这段代码和上面的公式推导，逐行理解。如果还不清楚，可以在 Python 中手动创建小矩阵（如 2×2）来验证每一步。

---

## 六、小结

| 概念 | 作用 |
|------|------|
| Q (Query) | 当前位置想要什么信息 |
| K (Key) | 每个位置能提供什么信息 |
| V (Value) | 每个位置实际携带的信息 |
| $QK^T$ | 计算 Q 和 K 的匹配程度 |
| $\div \sqrt{d_k}$ | 稳定训练，防止梯度消失 |
| softmax | 将分数转换为概率分布 |
| 加权求和 | 按概率综合所有位置的信息 |
| Mask | 强制模型忽略不该看的位置 |
