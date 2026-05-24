"""
tests/test_attention.py — 注意力机制正确性测试

验证注意力机制的数学正确性：
- 权重归一化（每行和为 1）
- 掩码功能（被掩蔽位置权重接近 0）
- 缩放效果

运行方法:
    pytest tests/test_attention.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.attention import scaled_dot_product_attention
from src.multi_head_attention import MultiHeadAttention


class TestAttentionCorrectness:
    """验证注意力机制的数学正确性。"""

    def test_weights_sum_to_one(self):
        """注意力权重每行之和应为 1（softmax 归一化）"""
        Q = torch.randn(2, 4, 8, 16)
        K = torch.randn(2, 4, 8, 16)
        V = torch.randn(2, 4, 8, 16)
        _, weights = scaled_dot_product_attention(Q, K, V)

        # 每行（最后一个维度）的和应该为 1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"注意力权重行之和不为 1！最大偏差: {(row_sums - 1).abs().max()}"

    def test_weights_non_negative(self):
        """注意力权重应全部非负（softmax 输出）"""
        Q = torch.randn(2, 4, 8, 16)
        K = torch.randn(2, 4, 8, 16)
        V = torch.randn(2, 4, 8, 16)
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert (weights >= 0).all(), "注意力权重存在负值！"

    def test_causal_mask_blocks_future(self):
        """因果掩码应阻止关注未来位置"""
        seq_len = 6
        Q = torch.randn(1, 1, seq_len, 16)
        K = torch.randn(1, 1, seq_len, 16)
        V = torch.randn(1, 1, seq_len, 16)

        # 下三角因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # 检查上三角部分（未来位置）的权重接近 0
        upper_triangle = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        future_weights = weights[0, 0][upper_triangle]
        assert future_weights.max() < 1e-5, \
            f"因果掩码失效！未来位置最大权重: {future_weights.max()}"

    def test_padding_mask_blocks_padded_positions(self):
        """Padding 掩码应阻止关注填充位置"""
        Q = torch.randn(1, 1, 6, 16)
        K = torch.randn(1, 1, 6, 16)
        V = torch.randn(1, 1, 6, 16)

        # 最后 2 个位置是 padding
        mask = torch.ones(1, 1, 1, 6).bool()
        mask[:, :, :, -2:] = False

        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 填充位置的权重应接近 0
        pad_weights = weights[0, 0, :, -2:]
        assert pad_weights.max() < 1e-5, \
            f"Padding 掩码失效！填充位置最大权重: {pad_weights.max()}"

    def test_identical_qk_produces_uniform_attention(self):
        """当 Q = K 且每个元素相同时，注意力应趋于均匀分布"""
        seq_len = 4
        d_k = 16
        # 所有位置的 Q 和 K 相同
        same_vec = torch.ones(1, 1, seq_len, d_k)
        V = torch.randn(1, 1, seq_len, d_k)

        _, weights = scaled_dot_product_attention(same_vec, same_vec, V)

        # 权重应该接近均匀分布（1/seq_len）
        expected = torch.ones(1, 1, seq_len, seq_len) / seq_len
        assert torch.allclose(weights, expected, atol=1e-5), \
            "相同 Q/K 时注意力应为均匀分布！"

    def test_scaling_effect(self):
        """缩放应该使 softmax 输出更平滑"""
        d_k = 64  # 较大的 d_k
        torch.manual_seed(42)
        Q = torch.randn(1, 1, 4, d_k)
        K = torch.randn(1, 1, 4, d_k)
        V = torch.randn(1, 1, 4, d_k)

        _, weights = scaled_dot_product_attention(Q, K, V)

        # 验证权重不全是 one-hot（说明缩放起了作用）
        # 由于缩放，softmax 应该产生相对平滑的分布，而不是极端的 one-hot
        min_weight = weights.min()
        assert min_weight > 1e-4, "缩放后最小权重不应太接近 0（分布应更平滑）"


class TestMultiHeadAttentionCorrectness:
    """验证多头注意力的正确性。"""

    def test_output_changes_with_different_inputs(self):
        """不同输入应产生不同输出"""
        mha = MultiHeadAttention(64, 4)
        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        out1, _ = mha(x1, x1, x1)
        out2, _ = mha(x2, x2, x2)
        assert not torch.allclose(out1, out2, atol=1e-3), \
            "不同输入应产生不同输出！"

    def test_deterministic_output(self):
        """eval 模式下，相同输入应产生相同输出"""
        mha = MultiHeadAttention(64, 4)
        mha.eval()
        x = torch.randn(1, 8, 64)
        out1, _ = mha(x, x, x)
        out2, _ = mha(x, x, x)
        assert torch.allclose(out1, out2), "相同输入应产生相同输出！"

    def test_d_model_not_divisible_by_heads(self):
        """d_model 不能被 num_heads 整除时应报错"""
        with pytest.raises(AssertionError):
            MultiHeadAttention(65, 4)  # 65 不能被 4 整除
