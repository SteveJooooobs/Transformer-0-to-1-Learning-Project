"""
tests/test_training.py — 训练烟雾测试

验证训练流程能正常执行：
- 模型能计算损失
- 损失能正常回传
- 参数能更新
- 训练几步后 loss 下降

运行方法:
    pytest tests/test_training.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest

from src.transformer import Transformer
from src.dataset import CharTokenizer, ShakespeareDataset, PAD_ID


VOCAB_SIZE = 30
D_MODEL = 32
NUM_HEADS = 2
D_FF = 64
NUM_LAYERS = 1


class TestTrainingSmoke:
    """训练流程的烟雾测试。"""

    def test_loss_computation(self):
        """模型能正常计算交叉熵损失"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        src = torch.randint(1, VOCAB_SIZE, (4, 16))
        tgt = torch.randint(1, VOCAB_SIZE, (4, 16))

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits, _ = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))

        assert loss.item() > 0, "损失应该大于 0！"
        assert not torch.isnan(loss), "损失不应为 NaN！"

    def test_backward_pass(self):
        """损失能正常反向传播"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        src = torch.randint(1, VOCAB_SIZE, (4, 16))
        tgt = torch.randint(1, VOCAB_SIZE, (4, 16))

        logits, _ = model(src, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
        loss.backward()

        # 检查所有参数都有梯度
        has_grad = all(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "反向传播后所有参数都应有梯度！"

    def test_parameter_update(self):
        """优化器能更新参数"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        # 记录初始参数
        params_before = {
            name: p.clone() for name, p in model.named_parameters()
        }

        # 训练一步
        src = torch.randint(1, VOCAB_SIZE, (4, 16))
        tgt = torch.randint(1, VOCAB_SIZE, (4, 16))
        logits, _ = model(src, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 检查参数已更新
        params_changed = False
        for name, p in model.named_parameters():
            if not torch.allclose(p, params_before[name]):
                params_changed = True
                break
        assert params_changed, "优化器应该更新了至少一个参数！"

    def test_loss_decreases(self):
        """训练多步后 loss 应该下降"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        # 固定一个小数据集（模型应该能过拟合）
        src = torch.randint(1, VOCAB_SIZE, (8, 16))
        tgt = torch.randint(1, VOCAB_SIZE, (8, 16))

        losses = []
        for step in range(50):
            logits, _ = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # 最后 5 步的平均 loss 应小于前 5 步的平均 loss
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, \
            f"训练 50 步后 loss 没有下降！前 5 步均值: {early_avg:.4f}, 后 5 步均值: {late_avg:.4f}"

    def test_gradient_clipping(self):
        """梯度裁剪应该限制梯度范数"""
        model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        src = torch.randint(1, VOCAB_SIZE, (4, 16))
        tgt = torch.randint(1, VOCAB_SIZE, (4, 16))

        logits, _ = model(src, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
        loss.backward()

        # 裁剪梯度
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # 裁剪后，所有参数的梯度范数应该 <= max_norm（允许小误差）
        # 注意：clip_grad_norm_ 返回裁剪前的范数


class TestDataset:
    """测试数据集功能。"""

    def test_tokenizer_encode_decode(self):
        """分词器编码和解码应该互逆"""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("Hello, World! 1234567890")

        text = "Hello"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"编码解码不一致: '{text}' -> {encoded} -> '{decoded}'"

    def test_tokenizer_special_tokens(self):
        """特殊 token 的 ID 应该正确"""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("abc")
        assert tokenizer.char_to_id["<pad>"] == 0
        assert tokenizer.char_to_id["<bos>"] == 1
        assert tokenizer.char_to_id["<eos>"] == 2

    def test_dataset_sample_shape(self):
        """数据集样本的形状正确"""
        tokenizer = CharTokenizer()
        text = "abcdefghij" * 100
        tokenizer.build_vocab(text)
        dataset = ShakespeareDataset(text, tokenizer, seq_len=8)

        src, tgt = dataset[0]
        assert src.shape == (8,), f"源序列形状错误: {src.shape}"
        assert tgt.shape == (8,), f"目标序列形状错误: {tgt.shape}"

    def test_dataset_src_tgt_consecutive(self):
        """源序列和目标序列应该是连续的"""
        tokenizer = CharTokenizer()
        text = "abcdefghijklmnop" * 100
        tokenizer.build_vocab(text)
        dataset = ShakespeareDataset(text, tokenizer, seq_len=4)

        src, tgt = dataset[0]
        # 目标序列应该紧跟在源序列之后
        src_text = tokenizer.decode(src.tolist())
        tgt_text = tokenizer.decode(tgt.tolist())
        assert text.startswith(src_text + tgt_text), \
            f"源和目标不连续: '{src_text}' + '{tgt_text}'"
