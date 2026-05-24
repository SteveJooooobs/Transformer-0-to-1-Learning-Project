"""
tests/test_shapes.py — 形状测试

验证所有模块的输入输出张量形状是否正确。
这是最基础的测试：如果形状不对，模型肯定跑不通。

运行方法:
    pytest tests/test_shapes.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.embedding import TokenEmbedding
from src.positional_encoding import PositionalEncoding
from src.attention import scaled_dot_product_attention
from src.multi_head_attention import MultiHeadAttention
from src.encoder import PositionwiseFeedForward, EncoderLayer, Encoder
from src.decoder import DecoderLayer, Decoder
from src.transformer import Transformer


# 通用测试参数
BATCH_SIZE = 2
SEQ_LEN = 10
D_MODEL = 64
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 2
VOCAB_SIZE = 50
D_K = D_MODEL // NUM_HEADS


class TestTokenEmbedding:
    """测试词嵌入层的输出形状。"""

    def test_output_shape(self):
        """嵌入层输出形状应为 [batch, seq_len, d_model]"""
        emb = TokenEmbedding(VOCAB_SIZE, D_MODEL)
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out = emb(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_different_seq_lengths(self):
        """不同序列长度应该都能正常工作"""
        emb = TokenEmbedding(VOCAB_SIZE, D_MODEL)
        for seq_len in [1, 5, 20, 100]:
            x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len))
            out = emb(x)
            assert out.shape == (BATCH_SIZE, seq_len, D_MODEL)


class TestPositionalEncoding:
    """测试位置编码的输出形状。"""

    def test_output_shape(self):
        """位置编码输出形状应与输入一致"""
        pe = PositionalEncoding(D_MODEL, max_len=5000, dropout=0.0)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out = pe(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_preserves_shape_for_various_lengths(self):
        """不同序列长度应保持形状不变"""
        pe = PositionalEncoding(D_MODEL, max_len=5000, dropout=0.0)
        for seq_len in [1, 10, 50, 200]:
            x = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
            out = pe(x)
            assert out.shape == (BATCH_SIZE, seq_len, D_MODEL)


class TestScaledDotProductAttention:
    """测试缩放点积注意力的输出形状。"""

    def test_output_shape(self):
        """注意力输出和权重形状"""
        Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
        K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
        V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
        assert weights.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)

    def test_different_qk_lengths(self):
        """Q 和 K 的序列长度可以不同"""
        q_len, k_len = 8, 12
        Q = torch.randn(BATCH_SIZE, NUM_HEADS, q_len, D_K)
        K = torch.randn(BATCH_SIZE, NUM_HEADS, k_len, D_K)
        V = torch.randn(BATCH_SIZE, NUM_HEADS, k_len, D_K)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (BATCH_SIZE, NUM_HEADS, q_len, D_K)
        assert weights.shape == (BATCH_SIZE, NUM_HEADS, q_len, k_len)


class TestMultiHeadAttention:
    """测试多头注意力的输出形状。"""

    def test_self_attention_shape(self):
        """自注意力输出形状"""
        mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output, weights = mha(x, x, x)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
        assert weights.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)

    def test_cross_attention_shape(self):
        """交叉注意力输出形状（Q 和 K/V 长度不同）"""
        mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
        q = torch.randn(BATCH_SIZE, 8, D_MODEL)
        kv = torch.randn(BATCH_SIZE, 12, D_MODEL)
        output, weights = mha(q, kv, kv)
        assert output.shape == (BATCH_SIZE, 8, D_MODEL)
        assert weights.shape == (BATCH_SIZE, NUM_HEADS, 8, 12)


class TestEncoder:
    """测试编码器的输出形状。"""

    def test_feedforward_shape(self):
        """前馈网络输出形状"""
        ffn = PositionwiseFeedForward(D_MODEL, D_FF)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out = ffn(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_encoder_layer_shape(self):
        """编码器层输出形状"""
        layer = EncoderLayer(D_MODEL, NUM_HEADS, D_FF)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out = layer(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_encoder_shape(self):
        """完整编码器输出形状"""
        encoder = Encoder(D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out = encoder(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


class TestDecoder:
    """测试解码器的输出形状。"""

    def test_decoder_layer_shape(self):
        """解码器层输出形状"""
        layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF)
        x = torch.randn(BATCH_SIZE, 8, D_MODEL)
        memory = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out, self_w, cross_w = layer(x, memory)
        assert out.shape == (BATCH_SIZE, 8, D_MODEL)
        assert self_w.shape == (BATCH_SIZE, NUM_HEADS, 8, 8)
        assert cross_w.shape == (BATCH_SIZE, NUM_HEADS, 8, SEQ_LEN)

    def test_decoder_shape(self):
        """完整解码器输出形状"""
        decoder = Decoder(D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        x = torch.randn(BATCH_SIZE, 8, D_MODEL)
        memory = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out, self_w, cross_w = decoder(x, memory)
        assert out.shape == (BATCH_SIZE, 8, D_MODEL)


class TestTransformer:
    """测试完整 Transformer 模型的输出形状。"""

    def test_output_shape(self):
        """Transformer 输出形状"""
        model = Transformer(VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
                           d_ff=D_FF, num_layers=NUM_LAYERS)
        src = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        tgt = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, 8))
        logits, cross_w = model(src, tgt)
        assert logits.shape == (BATCH_SIZE, 8, VOCAB_SIZE)
        assert cross_w.shape == (BATCH_SIZE, NUM_HEADS, 8, SEQ_LEN)

    def test_mask_shapes(self):
        """掩码形状正确"""
        model = Transformer(VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS)
        src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        tgt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 8))
        src_mask = model.make_src_mask(src)
        tgt_mask = model.make_tgt_mask(tgt)
        assert src_mask.shape == (BATCH_SIZE, 1, 1, SEQ_LEN)
        assert tgt_mask.shape == (BATCH_SIZE, 1, 8, 8)
