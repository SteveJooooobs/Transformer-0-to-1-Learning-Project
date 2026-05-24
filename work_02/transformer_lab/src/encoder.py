"""
encoder.py — 编码器（Encoder）

本模块实现了 Transformer 的编码器部分，包含：
1. PositionwiseFeedForward：位置无关的前馈网络
2. EncoderLayer：单层编码器（自注意力 + 前馈网络 + 残差连接 + 层归一化）
3. Encoder：由多层 EncoderLayer 堆叠组成的完整编码器

编码器的作用：
    接收源序列的嵌入表示，通过多层自注意力和前馈变换，
    生成包含全局上下文信息的特征表示（称为 Memory）。
    这些 Memory 特征会被传递给解码器，作为解码器交叉注意力的输入。

每个 EncoderLayer 的数据流：
    输入 x
    ↓
    [自注意力] → attn_out
    ↓
    [Dropout + 残差连接 + LayerNorm]: x = LayerNorm(x + Dropout(attn_out))
    ↓
    [前馈网络] → ff_out
    ↓
    [Dropout + 残差连接 + LayerNorm]: x = LayerNorm(x + Dropout(ff_out))
    ↓
    输出 x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_head_attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """
    位置无关的前馈网络（Position-wise Feed-Forward Network）。

    对序列中每个位置的特征向量独立地进行非线性变换。
    结构：Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model)

    直觉理解：
        如果说注意力层负责「收集信息」（让每个位置看到其他位置），
        那么前馈网络负责「处理信息」（对收集到的信息做非线性变换）。

    参数说明：
        d_ff 通常是 d_model 的 4 倍（本项目中 d_model=128, d_ff=512）。
        更大的 d_ff 意味着更强的表达能力，但也意味着更多的计算量。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈网络。

        参数:
            d_model (int): 输入和输出的特征维度。
            d_ff (int): 中间隐藏层的维度（通常为 d_model 的 4 倍）。
            dropout (float): Dropout 比率，默认 0.1。
        """
        super(PositionwiseFeedForward, self).__init__()

        # 第一个线性层：升维 d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        # 第二个线性层：降维 d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout 层：正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (Tensor): 输入特征
                形状: [batch_size, seq_len, d_model]

        返回:
            Tensor: 变换后的特征
                形状: [batch_size, seq_len, d_model]
        """
        # x: [B, L, d_model]
        # -> linear1 -> [B, L, d_ff] -> ReLU -> Dropout -> linear2 -> [B, L, d_model]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    单层编码器（Encoder Layer）。

    包含两个子层：
    1. 多头自注意力层（Multi-Head Self-Attention）
    2. 位置无关前馈网络（Position-wise Feed-Forward Network）

    每个子层都使用了：
    - 残差连接（Residual Connection）：output = x + SubLayer(x)
    - 层归一化（Layer Normalization）：稳定训练过程

    本实现使用 Post-LN 结构（先计算子层，再加残差，最后归一化），
    这是原始 Transformer 论文中使用的方式。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化编码器层。

        参数:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏层维度。
            dropout (float): Dropout 比率，默认 0.1。
        """
        super(EncoderLayer, self).__init__()

        # 子层 1：多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # 子层 2：前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 两个子层各自的 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 两个子层各自的 Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        前向传播。

        参数:
            x (Tensor): 输入特征
                形状: [batch_size, seq_len, d_model]
            src_mask (Tensor, 可选): 源序列的 Padding 掩码
                形状: [batch_size, 1, 1, seq_len]

        返回:
            Tensor: 编码器层输出
                形状: [batch_size, seq_len, d_model]
        """
        # ---- 子层 1：多头自注意力 ----
        # 自注意力：Q = K = V = x（每个位置都关注序列中的所有位置）
        # attn_output: [B, L, d_model]
        attn_output, _ = self.self_attn(x, x, x, src_mask)

        # 残差连接 + LayerNorm: x = LayerNorm(x + Dropout(attn_output))
        # x: [B, L, d_model]
        x = self.norm1(x + self.dropout1(attn_output))

        # ---- 子层 2：前馈网络 ----
        # ff_output: [B, L, d_model]
        ff_output = self.feed_forward(x)

        # 残差连接 + LayerNorm
        # x: [B, L, d_model]
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class Encoder(nn.Module):
    """
    完整编码器：由多个 EncoderLayer 堆叠组成。

    工作流程：
        输入嵌入向量 → EncoderLayer_1 → EncoderLayer_2 → ... → LayerNorm → Memory

    堆叠多层的意义：
        每增加一层，模型就能捕获更复杂的特征和更长距离的依赖关系。
        低层可能学习简单的局部模式，高层则学习抽象的全局模式。
    """

    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        """
        初始化编码器。

        参数:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏层维度。
            num_layers (int): 编码器层的堆叠数量。
            dropout (float): Dropout 比率，默认 0.1。
        """
        super(Encoder, self).__init__()

        # 使用 ModuleList 管理多个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终的 LayerNorm，稳定输出
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        """
        前向传播：将输入依次通过每一层编码器。

        参数:
            x (Tensor): 输入嵌入特征（已加位置编码）
                形状: [batch_size, seq_len, d_model]
            src_mask (Tensor, 可选): 源序列 Padding 掩码
                形状: [batch_size, 1, 1, seq_len]

        返回:
            Tensor: 编码器输出（Memory）
                形状: [batch_size, seq_len, d_model]
        """
        # 依次通过每一层编码器
        for layer in self.layers:
            # x: [B, L, d_model] -> [B, L, d_model]
            x = layer(x, src_mask)

        # 最终的 LayerNorm
        # x: [B, L, d_model]
        return self.norm(x)


if __name__ == "__main__":
    """模块自测：验证 Encoder 各组件的输入输出形状"""
    print("=" * 50)
    print("测试 Encoder 模块")
    print("=" * 50)

    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    batch_size = 2
    seq_len = 10

    # 测试前馈网络
    ffn = PositionwiseFeedForward(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    out = ffn(x)
    print(f"前馈网络输出形状: {out.shape}  (预期: [{batch_size}, {seq_len}, {d_model}])")
    assert out.shape == (batch_size, seq_len, d_model)
    print("✅ PositionwiseFeedForward 测试通过！")

    # 测试单层编码器
    enc_layer = EncoderLayer(d_model, num_heads, d_ff)
    out = enc_layer(x)
    print(f"编码器层输出形状: {out.shape}  (预期: [{batch_size}, {seq_len}, {d_model}])")
    assert out.shape == (batch_size, seq_len, d_model)
    print("✅ EncoderLayer 测试通过！")

    # 测试完整编码器
    encoder = Encoder(d_model, num_heads, d_ff, num_layers)
    out = encoder(x)
    print(f"编码器输出形状: {out.shape}  (预期: [{batch_size}, {seq_len}, {d_model}])")
    assert out.shape == (batch_size, seq_len, d_model)
    print("✅ Encoder 测试通过！")

    # 带掩码测试
    mask = torch.ones(batch_size, 1, 1, seq_len).bool()
    mask[:, :, :, -2:] = False  # 模拟最后 2 个位置是 padding
    out = encoder(x, mask)
    assert out.shape == (batch_size, seq_len, d_model)
    print("✅ 带掩码的 Encoder 测试通过！")

    print("✅ Encoder 全部测试通过！")
