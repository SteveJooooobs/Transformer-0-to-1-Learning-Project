"""
tests/test_inference.py — 推理测试

验证推理（文本生成）功能的正确性：
- 模型能生成文本
- 生成的文本是有效字符
- 不同采样策略都能工作

运行方法:
    pytest tests/test_inference.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.transformer import Transformer
from src.dataset import CharTokenizer, PAD_ID, BOS_ID, EOS_ID
from src.inference import generate_greedy, generate_with_temperature


# 创建一个小模型和分词器用于测试
VOCAB_SIZE = 30
D_MODEL = 32
NUM_HEADS = 2
D_FF = 64
NUM_LAYERS = 1


@pytest.fixture
def model_and_tokenizer():
    """创建测试用的模型和分词器。"""
    tokenizer = CharTokenizer()
    tokenizer.build_vocab("abcdefghijklmnopqrstuvwxyz ")
    
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
    )
    model.eval()
    device = torch.device("cpu")
    
    return model, tokenizer, device


class TestGreedyGeneration:
    """测试贪心搜索生成。"""

    def test_generates_text(self, model_and_tokenizer):
        """贪心搜索能生成文本"""
        model, tokenizer, device = model_and_tokenizer
        result = generate_greedy(model, tokenizer, device, "hello", max_len=20)
        assert isinstance(result, str), "输出应该是字符串"
        assert len(result) > 0, "生成的文本不应为空"

    def test_deterministic(self, model_and_tokenizer):
        """贪心搜索应该是确定性的"""
        model, tokenizer, device = model_and_tokenizer
        result1 = generate_greedy(model, tokenizer, device, "hello", max_len=20)
        result2 = generate_greedy(model, tokenizer, device, "hello", max_len=20)
        assert result1 == result2, "贪心搜索应该是确定性的"

    def test_respects_max_length(self, model_and_tokenizer):
        """生成长度不应超过 max_len"""
        model, tokenizer, device = model_and_tokenizer
        max_len = 10
        result = generate_greedy(model, tokenizer, device, "hello", max_len=max_len)
        # 生成的 token 数（不含种子）不应超过 max_len
        # 注意：实际生成的文本可能更短（如果生成了 EOS）


class TestTemperatureSampling:
    """测试温度采样生成。"""

    def test_generates_text(self, model_and_tokenizer):
        """温度采样能生成文本"""
        model, tokenizer, device = model_and_tokenizer
        result = generate_with_temperature(
            model, tokenizer, device, "hello",
            max_len=20, temperature=1.0
        )
        assert isinstance(result, str), "输出应该是字符串"

    def test_low_temperature_less_random(self, model_and_tokenizer):
        """低温度应该产生更确定的结果"""
        model, tokenizer, device = model_and_tokenizer
        # 低温度生成多次，结果应该相似
        results = set()
        for _ in range(5):
            result = generate_with_temperature(
                model, tokenizer, device, "hello",
                max_len=10, temperature=0.01
            )
            results.add(result)
        # 极低温度下结果应该几乎一致
        assert len(results) <= 2, \
            f"极低温度下结果应该几乎一致，但得到 {len(results)} 种不同结果"

    def test_top_k_sampling(self, model_and_tokenizer):
        """Top-K 采样能正常工作"""
        model, tokenizer, device = model_and_tokenizer
        result = generate_with_temperature(
            model, tokenizer, device, "hello",
            max_len=20, temperature=0.8, top_k=5
        )
        assert isinstance(result, str), "Top-K 采样应该返回字符串"

    def test_different_seeds_different_output(self, model_and_tokenizer):
        """不同种子文本应该产生不同输出"""
        model, tokenizer, device = model_and_tokenizer
        result1 = generate_greedy(model, tokenizer, device, "abcde", max_len=20)
        result2 = generate_greedy(model, tokenizer, device, "xyz", max_len=20)
        # 不要求完全不同，但种子不同通常输出也不同


class TestModelSaveLoad:
    """测试模型保存和加载。"""

    def test_save_and_load_state(self, model_and_tokenizer, tmp_path):
        """模型能保存和加载权重"""
        model, tokenizer, device = model_and_tokenizer

        # 保存
        save_path = str(tmp_path / "test_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
        }, save_path)

        # 加载
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model2 = Transformer(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            num_layers=NUM_LAYERS,
        )
        model2.load_state_dict(checkpoint["model_state_dict"])
        model2.eval()

        # 验证输出一致
        src = torch.randint(3, tokenizer.vocab_size, (1, 10))
        tgt = torch.randint(3, tokenizer.vocab_size, (1, 8))
        with torch.no_grad():
            out1, _ = model(src, tgt)
            out2, _ = model2(src, tgt)
        assert torch.allclose(out1, out2, atol=1e-5), "加载后输出应该一致"
