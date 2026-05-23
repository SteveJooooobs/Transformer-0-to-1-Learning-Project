# -*- coding: utf-8 -*-
"""
models 包：包含两种不同方式实现的 Transformer 模型模块。

此包提供以下核心模型类：
1. HandwrittenTransformer (位于 handwritten_transformer 模块)：
   纯手写实现（白盒化），从底层张量操作、多头注意力、位置编码、编码器层/解码器层一直到完整的 Seq2SeqTransformer。
2. TorchTransformer (位于 torch_transformer 模块)：
   使用 PyTorch 官方封装的底层模块（如 nn.TransformerEncoder, nn.TransformerDecoder 等）搭建的 Seq2Seq 架构。
"""

from .handwritten_transformer import HandwrittenTransformer
from .torch_transformer import TorchTransformer

__all__ = [
    'HandwrittenTransformer',
    'TorchTransformer'
]
