"""
multi_head_attention.py — 多头注意力机制（Multi-Head Attention）

本模块实现了 Transformer 的多头注意力层。它将输入的特征维度拆分为多个
「头」（head），每个头独立地学习不同的注意力模式，然后将结果拼接在一起。

核心思想：
    单个注意力头只能学习一种「关注模式」。例如，它可能只学会关注「相邻的词」。
    多头注意力让模型同时学习多种不同的关注模式：
    - 某个头可能关注「语法关系」（主语-谓语）
    - 某个头可能关注「位置邻近性」（相邻的词）
    - 某个头可能关注「语义相似性」（含义接近的词）

实现方式：
    1. 将 d_model 维输入通过线性变换，投影为 Q、K、V
    2. 将 Q、K、V 拆分为 num_heads 个子空间
    3. 在每个子空间中独立计算缩放点积注意力
    4. 将所有子空间的结果拼接起来
    5. 通过一个线性层输出最终结果
"""

import torch
import torch.nn as nn

from .attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    多头注意力层。

    它支持三种使用场景：
    1. 自注意力（Self-Attention）：Q = K = V = 同一个输入
    2. 交叉注意力（Cross-Attention）：Q 来自解码器，K 和 V 来自编码器
    3. 带掩码的自注意力：用于解码器中防止偷看未来的 token

    使用示例：
        >>> mha = MultiHeadAttention(d_model=128, num_heads=4)
        >>> x = torch.randn(2, 10, 128)  # [batch=2, seq_len=10, d_model=128]
        >>> output, weights = mha(x, x, x)  # 自注意力
        >>> output.shape  # torch.Size([2, 10, 128])
    """

    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力层。

        参数:
            d_model (int): 模型的隐藏维度。
            num_heads (int): 注意力头的数量。
                d_model 必须能被 num_heads 整除。

        异常:
            AssertionError: 如果 d_model 不能被 num_heads 整除。
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        # 每个头分配到的维度
        self.d_k = d_model // num_heads

        # Q、K、V 的线性投影层
        # 每个层将 d_model 维的输入投影为 d_model 维的输出
        # 实际上等价于 num_heads 个独立的 [d_model -> d_k] 投影的拼接
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出的线性投影层：将多头拼接的结果映射回 d_model 维
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        前向传播：计算多头注意力。

        参数:
            q (Tensor): 查询输入
                形状: [batch_size, seq_len_q, d_model]
            k (Tensor): 键输入
                形状: [batch_size, seq_len_k, d_model]
            v (Tensor): 值输入
                形状: [batch_size, seq_len_k, d_model]
            mask (Tensor, 可选): 掩码矩阵
                形状: 可广播至 [batch_size, num_heads, seq_len_q, seq_len_k]

        返回:
            tuple: (注意力输出, 注意力权重)
                - 输出: [batch_size, seq_len_q, d_model]
                - 权重: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)

        # ---- 第 1 步：线性投影 ----
        # 将输入通过线性层，得到 Q、K、V
        # [B, L, d_model] -> [B, L, d_model]
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # ---- 第 2 步：拆分为多个头 ----
        # [B, L, d_model] -> [B, L, num_heads, d_k] -> [B, num_heads, L, d_k]
        # 先 view 拆分最后一个维度，再 transpose 把 head 维度提前
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 此时 Q: [B, H, L_q, d_k], K: [B, H, L_k, d_k], V: [B, H, L_k, d_k]

        # ---- 第 3 步：计算注意力 ----
        # 调用 attention.py 中的 scaled_dot_product_attention
        # attn_output: [B, H, L_q, d_k]
        # attn_weights: [B, H, L_q, L_k]
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # ---- 第 4 步：拼接所有头 ----
        # [B, H, L_q, d_k] -> [B, L_q, H, d_k] -> [B, L_q, d_model]
        # transpose 后在内存中不连续，需要 .contiguous() 才能做 .view
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        # 此时 attn_output: [B, L_q, d_model]

        # ---- 第 5 步：输出投影 ----
        # [B, L_q, d_model] -> [B, L_q, d_model]
        output = self.W_o(attn_output)

        return output, attn_weights


if __name__ == "__main__":
    """模块自测：验证 MultiHeadAttention 的输入输出形状"""
    print("=" * 50)
    print("测试 MultiHeadAttention 模块")
    print("=" * 50)

    d_model = 128
    num_heads = 4
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 8

    mha = MultiHeadAttention(d_model, num_heads)

    # 自注意力测试（Q=K=V）
    x = torch.randn(batch_size, seq_len_q, d_model)
    output, weights = mha(x, x, x)
    print(f"自注意力输出形状: {output.shape}  (预期: [{batch_size}, {seq_len_q}, {d_model}])")
    assert output.shape == (batch_size, seq_len_q, d_model)
    print("✅ 自注意力测试通过！")

    # 交叉注意力测试（Q 和 K/V 长度不同）
    q = torch.randn(batch_size, seq_len_q, d_model)
    kv = torch.randn(batch_size, seq_len_k, d_model)
    output, weights = mha(q, kv, kv)
    print(f"交叉注意力输出形状: {output.shape}  (预期: [{batch_size}, {seq_len_q}, {d_model}])")
    print(f"交叉注意力权重形状: {weights.shape}  (预期: [{batch_size}, {num_heads}, {seq_len_q}, {seq_len_k}])")
    assert output.shape == (batch_size, seq_len_q, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    print("✅ 交叉注意力测试通过！")

    print("✅ MultiHeadAttention 全部测试通过！")
