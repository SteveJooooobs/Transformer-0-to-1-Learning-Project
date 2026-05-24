"""
transformer.py — 完整的 Seq2Seq Transformer 模型

本模块将所有组件组装成一个端到端的序列到序列（Seq2Seq）Transformer 模型。
它整合了：
- 词嵌入层（TokenEmbedding）
- 位置编码（PositionalEncoding）
- 编码器（Encoder）
- 解码器（Decoder）
- 线性输出层（Generator）

模型的数据流：
    源序列 → [词嵌入 + 位置编码] → [编码器] → Memory
    目标序列 → [词嵌入 + 位置编码] → [解码器(+Memory)] → [线性层] → 词表概率分布

任务说明：
    本项目使用 Tiny Shakespeare 数据集，任务是「给定前 N 个字符，预测接下来的 N 个字符」。
    这是一个字符级的 Seq2Seq 任务，源序列和目标序列共享同一个字符词表。
"""

import math
import torch
import torch.nn as nn

from .embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """
    完整的 Seq2Seq Transformer 模型。

    架构概览：
        ┌─────────────┐    ┌─────────────┐
        │  源序列 src  │    │ 目标序列 tgt │
        └──────┬──────┘    └──────┬──────┘
               ↓                  ↓
        ┌──────┴──────┐    ┌──────┴──────┐
        │  词嵌入层    │    │  词嵌入层    │
        │  + 位置编码  │    │  + 位置编码  │
        └──────┬──────┘    └──────┬──────┘
               ↓                  ↓
        ┌──────┴──────┐    ┌──────┴──────┐
        │   编码器     │───→│   解码器     │
        │  (Encoder)   │    │  (Decoder)   │
        └─────────────┘    └──────┬──────┘
                                  ↓
                           ┌──────┴──────┐
                           │  线性输出层  │
                           │ (Generator)  │
                           └──────┬──────┘
                                  ↓
                           词表概率分布 logits

    使用示例：
        >>> model = Transformer(vocab_size=65, d_model=128, num_heads=4,
        ...                     d_ff=512, num_layers=2, max_len=256)
        >>> src = torch.randint(0, 65, (2, 64))   # [batch=2, src_len=64]
        >>> tgt = torch.randint(0, 65, (2, 64))   # [batch=2, tgt_len=64]
        >>> logits, attn_weights = model(src, tgt)
        >>> logits.shape  # torch.Size([2, 64, 65])
    """

    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512,
                 num_layers=2, max_len=256, dropout=0.1, pad_id=0,
                 use_positional_encoding=True):
        """
        初始化 Transformer 模型。

        参数:
            vocab_size (int): 词表大小（源和目标共享词表）。
            d_model (int): 模型隐藏维度，默认 128。
            num_heads (int): 注意力头数，默认 4。
            d_ff (int): 前馈网络隐藏层维度，默认 512。
            num_layers (int): 编码器/解码器的层数，默认 2。
            max_len (int): 支持的最大序列长度，默认 256。
            dropout (float): Dropout 比率，默认 0.1。
            pad_id (int): 填充 token 的 ID，默认 0。
            use_positional_encoding (bool): 是否使用位置编码，默认 True。
                设为 False 可用于实验观察位置编码的影响。
        """
        super(Transformer, self).__init__()

        self.pad_id = pad_id
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        # ---- 嵌入层：源序列和目标序列共享同一个嵌入层 ----
        # 因为本项目中源和目标使用相同的字符词表
        self.embedding = TokenEmbedding(vocab_size, d_model)

        # ---- 位置编码：源序列和目标序列共享 ----
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # ---- 编码器 ----
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)

        # ---- 解码器 ----
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # ---- 输出层（Generator）----
        # 将 d_model 维特征映射到词表大小，用于预测下一个 token
        self.generator = nn.Linear(d_model, vocab_size)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """
        使用 Xavier 均匀初始化所有可训练参数。

        Xavier 初始化能确保信号在前向传播和反向传播中
        保持合理的方差，有助于训练稳定性。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """
        构建源序列的 Padding 掩码。

        功能：标记源序列中哪些位置是填充（PAD），这些位置不应被注意力关注。

        参数:
            src (Tensor): 源序列 token ID
                形状: [batch_size, src_len]

        返回:
            Tensor: 布尔掩码，True 表示有效位置，False 表示填充位置
                形状: [batch_size, 1, 1, src_len]
                扩展的维度用于与注意力得分矩阵 [B, H, L_q, L_k] 自动广播
        """
        # (src != pad_id) -> [B, src_len] 布尔张量
        # .unsqueeze(1).unsqueeze(2) -> [B, 1, 1, src_len]
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        构建目标序列的复合掩码（Padding 掩码 + 因果掩码）。

        功能：
        1. Padding 掩码：忽略填充位置
        2. 因果掩码（下三角矩阵）：防止解码器偷看未来的 token

        两个掩码通过逻辑与（AND）组合。

        参数:
            tgt (Tensor): 目标序列 token ID
                形状: [batch_size, tgt_len]

        返回:
            Tensor: 复合掩码
                形状: [batch_size, 1, tgt_len, tgt_len]
        """
        tgt_len = tgt.size(1)

        # 1. Padding 掩码: [B, 1, 1, tgt_len]
        tgt_pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)

        # 2. 因果掩码（下三角矩阵）: [tgt_len, tgt_len]
        #    位置 i 只能关注 0..i 的位置（不能看未来）
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).bool()
        # 扩展维度: [1, 1, tgt_len, tgt_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        # 3. 组合：两个掩码逻辑与
        # tgt_mask: [B, 1, tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & causal_mask

        return tgt_mask

    def forward(self, src, tgt):
        """
        前向传播：从源序列和目标序列计算输出概率分布。

        参数:
            src (Tensor): 源序列 token ID
                形状: [batch_size, src_len]
            tgt (Tensor): 目标序列 token ID（训练时为右移的真实目标）
                形状: [batch_size, tgt_len]

        返回:
            tuple: (logits, cross_attention_weights)
                - logits: 词表上的未归一化概率分布
                    形状: [batch_size, tgt_len, vocab_size]
                - cross_attention_weights: 最后一层解码器的交叉注意力权重
                    形状: [batch_size, num_heads, tgt_len, src_len]
        """
        # ---- 第 1 步：构造掩码 ----
        # src_mask: [B, 1, 1, src_len]
        src_mask = self.make_src_mask(src)
        # tgt_mask: [B, 1, tgt_len, tgt_len]
        tgt_mask = self.make_tgt_mask(tgt)
        # cross_mask: 与 src_mask 相同，用于交叉注意力
        cross_mask = src_mask

        # ---- 第 2 步：源序列嵌入 + 位置编码 ----
        # src_emb: [B, src_len, d_model]
        src_emb = self.embedding(src)
        if self.use_positional_encoding:
            src_emb = self.pos_encoding(src_emb)

        # ---- 第 3 步：目标序列嵌入 + 位置编码 ----
        # tgt_emb: [B, tgt_len, d_model]
        tgt_emb = self.embedding(tgt)
        if self.use_positional_encoding:
            tgt_emb = self.pos_encoding(tgt_emb)

        # ---- 第 4 步：编码器 ----
        # memory: [B, src_len, d_model]
        memory = self.encoder(src_emb, src_mask)

        # ---- 第 5 步：解码器 ----
        # dec_output: [B, tgt_len, d_model]
        dec_output, _, cross_weights = self.decoder(
            tgt_emb, memory, tgt_mask, cross_mask
        )

        # ---- 第 6 步：线性输出层 ----
        # logits: [B, tgt_len, vocab_size]
        logits = self.generator(dec_output)

        return logits, cross_weights


if __name__ == "__main__":
    """模块自测：验证完整 Transformer 模型的输入输出"""
    print("=" * 50)
    print("测试完整 Transformer 模型")
    print("=" * 50)

    vocab_size = 65
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    batch_size = 2
    src_len = 64
    tgt_len = 64

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=256,
        dropout=0.1,
        pad_id=0
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 模拟输入
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))

    # 前向传播
    logits, cross_weights = model(src, tgt)
    print(f"\n输出 logits 形状: {logits.shape}  (预期: [{batch_size}, {tgt_len}, {vocab_size}])")
    print(f"交叉注意力权重形状: {cross_weights.shape}  (预期: [{batch_size}, {num_heads}, {tgt_len}, {src_len}])")

    assert logits.shape == (batch_size, tgt_len, vocab_size), "logits 形状不匹配！"
    assert cross_weights.shape == (batch_size, num_heads, tgt_len, src_len), "注意力权重形状不匹配！"

    # 测试掩码
    src_with_pad = src.clone()
    src_with_pad[:, -5:] = 0  # 最后 5 个位置设为 padding
    logits2, _ = model(src_with_pad, tgt)
    assert logits2.shape == (batch_size, tgt_len, vocab_size)
    print("✅ 带 Padding 的前向传播测试通过！")

    print("✅ Transformer 模型全部测试通过！")
