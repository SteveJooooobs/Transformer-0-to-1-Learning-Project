"""
positional_encoding.py — 正弦/余弦位置编码（Positional Encoding）

本模块实现了 Transformer 论文（"Attention Is All You Need"）中提出的
正弦/余弦位置编码。由于 Transformer 使用自注意力而非循环结构，模型本身
无法感知 token 在序列中的位置顺序。位置编码通过向嵌入向量中注入位置信息
来解决这个问题。

数学公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
    - pos 是 token 在序列中的位置（0, 1, 2, ...）
    - i 是维度索引（0, 1, 2, ..., d_model/2 - 1）
    - d_model 是嵌入维度

直觉理解：
    不同频率的正弦/余弦波叠加在一起，类似于给每个位置分配了一个独特的
    「指纹」。低频维度编码远距离的位置关系，高频维度编码近距离的位置关系。
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码层。

    功能：
        为输入的嵌入向量叠加位置信息，使模型能够区分不同位置的 token。
        位置编码矩阵在初始化时预计算，不参与梯度更新。

    使用示例：
        >>> pe = PositionalEncoding(d_model=128, max_len=5000, dropout=0.1)
        >>> x = torch.randn(2, 10, 128)  # [batch=2, seq_len=10, d_model=128]
        >>> out = pe(x)
        >>> out.shape  # torch.Size([2, 10, 128])
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码矩阵。

        参数:
            d_model (int): 嵌入向量的维度（必须为偶数）。
            max_len (int): 支持的最大序列长度，默认 5000。
            dropout (float): Dropout 比率，用于正则化，默认 0.1。
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # ---- 预计算位置编码矩阵 ----

        # 创建一个全零矩阵，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 生成位置索引：[0, 1, 2, ..., max_len-1]，形状 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算频率的分母项 div_term，形状 [d_model/2]
        # 公式：10000^(2i/d_model) = exp(2i * (-ln(10000)/d_model))
        # 使用 exp-log 技巧保证数值稳定性
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度使用 sin，奇数维度使用 cos
        # pe[:, 0::2] 取所有偶数列（第 0, 2, 4, ... 列）
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] 取所有奇数列（第 1, 3, 5, ... 列）
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度：[max_len, d_model] -> [1, max_len, d_model]
        # 这样可以直接与 [batch_size, seq_len, d_model] 的输入进行广播相加
        pe = pe.unsqueeze(0)

        # 注册为 buffer：它是模型状态的一部分，会随模型一起保存/加载，
        # 但不会被优化器更新（不需要梯度）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播：将位置编码加到输入的嵌入向量上。

        参数:
            x (Tensor): 输入的嵌入向量（已经过缩放）
                形状: [batch_size, seq_len, d_model]

        返回:
            Tensor: 叠加位置编码后的向量
                形状: [batch_size, seq_len, d_model]
        """
        # 从预计算的 pe 矩阵中截取当前序列长度对应的部分
        # self.pe[:, :x.size(1)] 形状: [1, seq_len, d_model]
        # 广播相加后形状不变: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]

        # Dropout 正则化，随机置零部分维度，防止过拟合
        return self.dropout(x)


if __name__ == "__main__":
    """模块自测：验证 PositionalEncoding 的输入输出形状"""
    print("=" * 50)
    print("测试 PositionalEncoding 模块")
    print("=" * 50)

    d_model = 128
    batch_size = 2
    seq_len = 10

    pe = PositionalEncoding(d_model=d_model, max_len=5000, dropout=0.1)

    # 模拟输入：已经过嵌入层的向量
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}  (预期: [{batch_size}, {seq_len}, {d_model}])")

    output = pe(x)
    print(f"输出形状: {output.shape}  (预期: [{batch_size}, {seq_len}, {d_model}])")

    assert output.shape == (batch_size, seq_len, d_model), "形状不匹配！"

    # 验证位置编码值不为全零
    pe_values = pe.pe[0, :5, :4]
    print(f"\n位置编码矩阵（前5行，前4列）:\n{pe_values}")
    assert pe_values.abs().sum() > 0, "位置编码不应为全零！"

    print("✅ PositionalEncoding 测试通过！")
