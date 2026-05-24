"""
attention.py — 缩放点积注意力（Scaled Dot-Product Attention）

本模块实现了 Transformer 的核心运算：缩放点积注意力机制。
它是多头注意力的基础构建块，负责计算 Query 与 Key 之间的相似度，
然后用这个相似度作为权重对 Value 进行加权求和。

核心公式：
    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

直觉理解：
    想象你在图书馆查书——
    - Query（查询）= 你心中想找的主题
    - Key（键）= 每本书封面上的标签
    - Value（值）= 书的实际内容
    - 注意力权重 = 你的查询与每本书标签的匹配程度
    - 最终输出 = 根据匹配程度，对所有书的内容做加权组合
"""

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    计算缩放点积注意力。

    这是 Transformer 中最核心的计算步骤。它让模型能够「关注」输入序列中
    与当前位置最相关的部分。

    参数:
        Q (Tensor): 查询矩阵（Query）
            形状: [batch_size, num_heads, seq_len_q, d_k]
        K (Tensor): 键矩阵（Key）
            形状: [batch_size, num_heads, seq_len_k, d_k]
        V (Tensor): 值矩阵（Value）
            形状: [batch_size, num_heads, seq_len_k, d_k]
        mask (Tensor, 可选): 掩码矩阵，用于屏蔽某些位置
            形状: 可广播至 [batch_size, num_heads, seq_len_q, seq_len_k]
            约定: mask 中值为 False（或 0）的位置将被屏蔽（设为负无穷）

    返回:
        tuple: (注意力输出, 注意力权重)
            - 注意力输出: [batch_size, num_heads, seq_len_q, d_k]
            - 注意力权重: [batch_size, num_heads, seq_len_q, seq_len_k]

    计算步骤详解:
        1. Q 和 K 做点积 -> 得到原始注意力得分
        2. 除以 √d_k -> 缩放，防止得分过大导致 softmax 梯度消失
        3. 应用 mask -> 将不应关注的位置得分设为 -∞
        4. softmax 归一化 -> 得到注意力权重（和为 1 的概率分布）
        5. 权重乘以 V -> 加权求和得到最终输出
    """
    # 获取每个头的维度 d_k，用于缩放
    d_k = Q.size(-1)

    # ---- 第 1 步：计算注意力得分 ----
    # Q: [B, H, L_q, d_k]，K^T: [B, H, d_k, L_k]
    # 矩阵乘法后得到: [B, H, L_q, L_k]
    # 每个元素表示 Query 中的一个位置与 Key 中的一个位置的「相似度」
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # ---- 第 2 步：缩放 ----
    # 除以 √d_k，防止当 d_k 很大时，点积值过大
    # 原因：如果 Q 和 K 的元素都是均值 0、方差 1 的随机变量，
    #       它们的点积的方差为 d_k。除以 √d_k 使方差回到 1，
    #       从而让 softmax 的输入保持在合理范围内。
    # scores: [B, H, L_q, L_k]
    scores = scores / math.sqrt(d_k)

    # ---- 第 3 步：应用掩码 ----
    if mask is not None:
        # 将 mask 中为 False 的位置对应的得分设为极大负数（-1e9）
        # 这样经过 softmax 后，这些位置的权重趋近于 0
        # 用途 1: Padding Mask — 忽略填充 token
        # 用途 2: Causal Mask — 防止解码器偷看未来的 token
        scores = scores.masked_fill(mask == False, -1e9)

    # ---- 第 4 步：Softmax 归一化 ----
    # 沿最后一个维度（L_k）做 softmax，使每一行的权重之和为 1
    # attn_weights: [B, H, L_q, L_k]
    # 每个值表示：当前 Query 位置应该给对应 Key 位置多少「关注度」
    attn_weights = F.softmax(scores, dim=-1)

    # ---- 第 5 步：加权求和 ----
    # attn_weights: [B, H, L_q, L_k] × V: [B, H, L_k, d_k]
    # 结果: [B, H, L_q, d_k]
    # 每个 Query 位置的输出 = 所有 Value 向量的加权组合
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


if __name__ == "__main__":
    """模块自测：验证 scaled_dot_product_attention 的正确性"""
    print("=" * 50)
    print("测试 Scaled Dot-Product Attention")
    print("=" * 50)

    batch_size = 2
    num_heads = 4
    seq_len = 6
    d_k = 32

    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_k)

    # 无掩码测试
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"输出形状: {output.shape}  (预期: [{batch_size}, {num_heads}, {seq_len}, {d_k}])")
    print(f"权重形状: {weights.shape}  (预期: [{batch_size}, {num_heads}, {seq_len}, {seq_len}])")

    assert output.shape == (batch_size, num_heads, seq_len, d_k), "输出形状不匹配！"
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), "权重形状不匹配！"

    # 验证注意力权重每行之和为 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "注意力权重之和应为 1！"
    print("✅ 注意力权重归一化验证通过！")

    # 带掩码测试（因果掩码）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    # 验证掩码位置的权重趋近于 0
    upper_weights = weights_masked[:, :, 0, 1:]  # 第 0 行不应关注后面的位置
    assert upper_weights.max() < 1e-5, "因果掩码未生效：不应关注的位置权重应接近 0！"
    print("✅ 因果掩码验证通过！")

    print("✅ Scaled Dot-Product Attention 全部测试通过！")
