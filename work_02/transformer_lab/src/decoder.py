"""
decoder.py — 解码器（Decoder）

本模块实现了 Transformer 的解码器部分，包含：
1. DecoderLayer：单层解码器
2. Decoder：由多层 DecoderLayer 堆叠组成的完整解码器

解码器的作用：
    接收目标序列的嵌入表示和编码器输出的 Memory，逐步生成输出序列。
    解码器的关键特点是使用因果掩码（Causal Mask），确保在预测第 t 个
    位置时只能看到前 t-1 个位置的信息，不能「偷看」未来的 token。

每个 DecoderLayer 的数据流（比 EncoderLayer 多一个子层）：
    输入 x + Memory（来自编码器）
    ↓
    [掩码自注意力] → 只关注已生成的部分
    ↓
    [Dropout + 残差连接 + LayerNorm]
    ↓
    [交叉注意力] → Q 来自解码器，K/V 来自编码器 Memory
    ↓
    [Dropout + 残差连接 + LayerNorm]
    ↓
    [前馈网络]
    ↓
    [Dropout + 残差连接 + LayerNorm]
    ↓
    输出 x
"""

import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .encoder import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    单层解码器（Decoder Layer）。

    包含三个子层：
    1. 掩码多头自注意力（Masked Multi-Head Self-Attention）
       - Q = K = V = 解码器输入
       - 使用因果掩码，防止偷看未来的 token
    2. 多头交叉注意力（Multi-Head Cross-Attention）
       - Q = 解码器特征，K = V = 编码器的 Memory
       - 让解码器能够「查询」编码器的输出
    3. 位置无关前馈网络（Position-wise Feed-Forward Network）
       - 对每个位置独立地做非线性变换
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化解码器层。

        参数:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏层维度。
            dropout (float): Dropout 比率，默认 0.1。
        """
        super(DecoderLayer, self).__init__()

        # 子层 1：掩码自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # 子层 2：交叉注意力（解码器查询编码器）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        # 子层 3：前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 三个子层各自的 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 三个子层各自的 Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, cross_mask=None):
        """
        前向传播。

        参数:
            x (Tensor): 解码器输入（目标序列的嵌入特征）
                形状: [batch_size, tgt_len, d_model]
            memory (Tensor): 编码器输出（Memory）
                形状: [batch_size, src_len, d_model]
            tgt_mask (Tensor, 可选): 目标序列掩码（包含因果掩码 + Padding 掩码）
                形状: [batch_size, 1, tgt_len, tgt_len]
            cross_mask (Tensor, 可选): 交叉注意力掩码（源序列的 Padding 掩码）
                形状: [batch_size, 1, 1, src_len]

        返回:
            tuple: (解码器层输出, 自注意力权重, 交叉注意力权重)
                - 输出: [batch_size, tgt_len, d_model]
                - 自注意力权重: [batch_size, num_heads, tgt_len, tgt_len]
                - 交叉注意力权重: [batch_size, num_heads, tgt_len, src_len]
        """
        # ---- 子层 1：掩码自注意力 ----
        # Q = K = V = x，使用 tgt_mask 防止偷看未来
        # self_attn_output: [B, L_tgt, d_model]
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)

        # 残差连接 + LayerNorm
        # x: [B, L_tgt, d_model]
        x = self.norm1(x + self.dropout1(self_attn_output))

        # ---- 子层 2：交叉注意力 ----
        # Q = x（来自解码器），K = V = memory（来自编码器）
        # 这一步让解码器能够「查看」源序列的信息
        # cross_attn_output: [B, L_tgt, d_model]
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, memory, memory, cross_mask
        )

        # 残差连接 + LayerNorm
        # x: [B, L_tgt, d_model]
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # ---- 子层 3：前馈网络 ----
        # ff_output: [B, L_tgt, d_model]
        ff_output = self.feed_forward(x)

        # 残差连接 + LayerNorm
        # x: [B, L_tgt, d_model]
        x = self.norm3(x + self.dropout3(ff_output))

        return x, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    """
    完整解码器：由多个 DecoderLayer 堆叠组成。

    工作流程：
        目标嵌入向量 + Memory → DecoderLayer_1 → DecoderLayer_2 → ... → LayerNorm → 输出

    解码过程说明：
        - 训练时：使用 Teacher Forcing，一次性输入整个目标序列
        - 推理时：逐步生成，每次只生成一个 token，然后作为下一步的输入
    """

    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        """
        初始化解码器。

        参数:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏层维度。
            num_layers (int): 解码器层的堆叠数量。
            dropout (float): Dropout 比率，默认 0.1。
        """
        super(Decoder, self).__init__()

        # 使用 ModuleList 管理多个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终的 LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, cross_mask=None):
        """
        前向传播：将输入依次通过每一层解码器。

        参数:
            x (Tensor): 目标序列嵌入特征（已加位置编码）
                形状: [batch_size, tgt_len, d_model]
            memory (Tensor): 编码器输出
                形状: [batch_size, src_len, d_model]
            tgt_mask (Tensor, 可选): 目标序列掩码
            cross_mask (Tensor, 可选): 交叉注意力掩码

        返回:
            tuple: (解码器输出, 最后一层的自注意力权重, 最后一层的交叉注意力权重)
                - 输出: [batch_size, tgt_len, d_model]
                - 自注意力权重: [batch_size, num_heads, tgt_len, tgt_len]
                - 交叉注意力权重: [batch_size, num_heads, tgt_len, src_len]
        """
        # 保存最后一层的注意力权重，用于可视化
        last_self_attn_weights = None
        last_cross_attn_weights = None

        # 依次通过每一层解码器
        for layer in self.layers:
            # x: [B, L_tgt, d_model] -> [B, L_tgt, d_model]
            x, last_self_attn_weights, last_cross_attn_weights = layer(
                x, memory, tgt_mask, cross_mask
            )

        # 最终的 LayerNorm
        # x: [B, L_tgt, d_model]
        return self.norm(x), last_self_attn_weights, last_cross_attn_weights


if __name__ == "__main__":
    """模块自测：验证 Decoder 各组件的输入输出形状"""
    print("=" * 50)
    print("测试 Decoder 模块")
    print("=" * 50)

    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    batch_size = 2
    tgt_len = 8
    src_len = 10

    # 模拟输入
    x = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)

    # 构造因果掩码
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
    tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L_tgt, L_tgt]

    # 测试单层解码器
    dec_layer = DecoderLayer(d_model, num_heads, d_ff)
    out, self_w, cross_w = dec_layer(x, memory, tgt_mask)
    print(f"解码器层输出形状: {out.shape}  (预期: [{batch_size}, {tgt_len}, {d_model}])")
    print(f"自注意力权重形状: {self_w.shape}  (预期: [{batch_size}, {num_heads}, {tgt_len}, {tgt_len}])")
    print(f"交叉注意力权重形状: {cross_w.shape}  (预期: [{batch_size}, {num_heads}, {tgt_len}, {src_len}])")
    assert out.shape == (batch_size, tgt_len, d_model)
    print("✅ DecoderLayer 测试通过！")

    # 测试完整解码器
    decoder = Decoder(d_model, num_heads, d_ff, num_layers)
    out, self_w, cross_w = decoder(x, memory, tgt_mask)
    print(f"解码器输出形状: {out.shape}  (预期: [{batch_size}, {tgt_len}, {d_model}])")
    assert out.shape == (batch_size, tgt_len, d_model)
    print("✅ Decoder 测试通过！")

    print("✅ Decoder 全部测试通过！")
