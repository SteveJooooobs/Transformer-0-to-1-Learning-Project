"""
tests/test_forward.py — 前向传播测试

验证所有模块的前向传播能正常执行，不抛出异常。
包括：单个模块、带掩码、不同批次大小等场景。

运行方法:
    pytest tests/test_forward.py -v
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


D_MODEL = 64
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 2
VOCAB_SIZE = 50


class TestForwardPass:
    """测试各模块的前向传播能正常执行。"""

    def test_embedding_forward(self):
        """TokenEmbedding 前向传播"""
        emb = TokenEmbedding(VOCAB_SIZE, D_MODEL)
        x = torch.randint(0, VOCAB_SIZE, (2, 10))
        out = emb(x)
        assert out.requires_grad  # 输出应该有梯度

    def test_positional_encoding_forward(self):
        """PositionalEncoding 前向传播"""
        pe = PositionalEncoding(D_MODEL, dropout=0.0)
        x = torch.randn(2, 10, D_MODEL)
        out = pe(x)
        # 验证位置编码确实改变了输入
        assert not torch.allclose(x, out)

    def test_attention_with_mask(self):
        """带掩码的注意力前向传播"""
        B, H, L, dk = 2, 4, 8, 16
        Q = torch.randn(B, H, L, dk)
        K = torch.randn(B, H, L, dk)
        V = torch.randn(B, H, L, dk)
        mask = torch.tril(torch.ones(L, L)).bool().unsqueeze(0).unsqueeze(0)
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        assert not torch.isnan(output).any()

    def test_multi_head_attention_with_mask(self):
        """带掩码的多头注意力前向传播"""
        mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
        x = torch.randn(2, 10, D_MODEL)
        mask = torch.ones(2, 1, 1, 10).bool()
        mask[:, :, :, -3:] = False
        output, weights = mha(x, x, x, mask)
        assert not torch.isnan(output).any()

    def test_encoder_with_mask(self):
        """带掩码的编码器前向传播"""
        encoder = Encoder(D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        x = torch.randn(2, 10, D_MODEL)
        mask = torch.ones(2, 1, 1, 10).bool()
        mask[:, :, :, -2:] = False
        out = encoder(x, mask)
        assert not torch.isnan(out).any()

    def test_decoder_with_masks(self):
        """带掩码的解码器前向传播"""
        decoder = Decoder(D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        x = torch.randn(2, 8, D_MODEL)
        memory = torch.randn(2, 10, D_MODEL)

        tgt_mask = torch.tril(torch.ones(8, 8)).bool().unsqueeze(0).unsqueeze(0)
        cross_mask = torch.ones(2, 1, 1, 10).bool()

        out, _, _ = decoder(x, memory, tgt_mask, cross_mask)
        assert not torch.isnan(out).any()

    def test_full_transformer_forward(self):
        """完整 Transformer 前向传播"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        src = torch.randint(1, VOCAB_SIZE, (2, 10))
        tgt = torch.randint(1, VOCAB_SIZE, (2, 8))
        logits, weights = model(src, tgt)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(weights).any()

    def test_transformer_with_padding(self):
        """Transformer 处理含 padding 的输入"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        src = torch.randint(1, VOCAB_SIZE, (2, 10))
        tgt = torch.randint(1, VOCAB_SIZE, (2, 8))
        src[:, -3:] = 0  # 源序列末尾 padding
        tgt[:, -2:] = 0  # 目标序列末尾 padding
        logits, weights = model(src, tgt)
        assert not torch.isnan(logits).any()

    def test_batch_size_one(self):
        """batch_size=1 的情况"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        src = torch.randint(1, VOCAB_SIZE, (1, 10))
        tgt = torch.randint(1, VOCAB_SIZE, (1, 8))
        logits, _ = model(src, tgt)
        assert logits.shape == (1, 8, VOCAB_SIZE)

    def test_seq_len_one(self):
        """序列长度为 1 的情况"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        src = torch.randint(1, VOCAB_SIZE, (2, 1))
        tgt = torch.randint(1, VOCAB_SIZE, (2, 1))
        logits, _ = model(src, tgt)
        assert logits.shape == (2, 1, VOCAB_SIZE)

    def test_gradient_flow(self):
        """验证梯度能正常回传"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        src = torch.randint(1, VOCAB_SIZE, (2, 10))
        tgt = torch.randint(1, VOCAB_SIZE, (2, 8))
        logits, _ = model(src, tgt)
        loss = logits.sum()
        loss.backward()

        # 检查所有参数都有梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度！"
