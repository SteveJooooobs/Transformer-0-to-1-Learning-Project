"""
embedding.py — 词嵌入层（Token Embedding）

本模块实现了 Transformer 中的词嵌入层，负责将离散的 token ID 映射为
连续的高维向量表示。嵌入后的向量会乘以 √d_model 进行缩放，防止在
加上位置编码后被其数值掩盖。

核心概念：
    - Token（词元）是文本的最小单位（本项目中是单个字符）
    - 每个 Token 对应词表中的一个整数 ID
    - Embedding 层将这些整数 ID 转换为 d_model 维的浮点向量
    - 这些向量在训练过程中会被不断优化，学习到语义信息
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    词嵌入层：将 token ID 映射为 d_model 维的连续向量。

    原理说明：
        在自然语言处理中，我们不能直接将离散的整数 ID 输入神经网络。
        Embedding 层本质上是一个「查找表」（Lookup Table），每个 token ID
        对应表中的一行向量。这些向量在训练过程中不断更新，最终学会编码
        语义信息——相似含义的 token 会拥有相似的向量表示。

    使用示例：
        >>> embedding = TokenEmbedding(vocab_size=65, d_model=128)
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # [batch=2, seq_len=3]
        >>> out = embedding(x)
        >>> out.shape  # torch.Size([2, 3, 128])
    """

    def __init__(self, vocab_size, d_model):
        """
        初始化词嵌入层。

        参数:
            vocab_size (int): 词表大小，即不同 token 的总数。
            d_model (int): 嵌入向量的维度，也是整个 Transformer 的隐藏维度。
        """
        super(TokenEmbedding, self).__init__()

        # nn.Embedding 内部维护一个 [vocab_size, d_model] 的权重矩阵
        # 每次 forward 时，根据输入的 token ID 从中「查表」取出对应的行向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 保存 d_model，用于后续的缩放操作
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播：将 token ID 转换为嵌入向量，并进行缩放。

        参数:
            x (Tensor): 输入的 token ID 张量
                形状: [batch_size, seq_len]
                类型: torch.long（整数）

        返回:
            Tensor: 缩放后的嵌入向量
                形状: [batch_size, seq_len, d_model]

        缩放说明:
            乘以 √d_model 是因为嵌入向量的初始值通常较小（均值接近 0，
            方差接近 1/d_model），而位置编码的值域在 [-1, 1]。如果不缩放，
            位置编码的信号会相对过强，盖过了嵌入向量携带的语义信息。
        """
        # x: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        embedded = self.embedding(x)

        # 缩放：乘以 √d_model，平衡嵌入向量与位置编码的数值量级
        # scaled: [batch_size, seq_len, d_model]
        scaled = embedded * math.sqrt(self.d_model)

        return scaled


if __name__ == "__main__":
    """模块自测：验证 TokenEmbedding 的输入输出形状"""
    print("=" * 50)
    print("测试 TokenEmbedding 模块")
    print("=" * 50)

    vocab_size = 65
    d_model = 128
    batch_size = 2
    seq_len = 10

    embedding = TokenEmbedding(vocab_size, d_model)

    # 模拟输入：batch_size=2, seq_len=10 的 token ID
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入形状: {x.shape}  (预期: [{batch_size}, {seq_len}])")

    output = embedding(x)
    print(f"输出形状: {output.shape}  (预期: [{batch_size}, {seq_len}, {d_model}])")

    assert output.shape == (batch_size, seq_len, d_model), "形状不匹配！"
    print("✅ TokenEmbedding 测试通过！")
