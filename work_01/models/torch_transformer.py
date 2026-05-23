# -*- coding: utf-8 -*-
"""
官方封装 Transformer 模块：models/torch_transformer.py
本模块使用 PyTorch 官方的高级封装 API (nn.TransformerEncoder, nn.TransformerDecoder) 构建 Seq2Seq 模型。
核心要点：
1. 共享相同的 Embedding 和自写正弦/余弦位置编码 (PositionalEncoding) 以保证对比的公平性。
2. 使用官方内置的注意力与前馈机制，自动使能 PyTorch 底层的优化能力（如 FlashAttention 等 C++ 加速核心）。
3. 实现了一个优雅的 Forward Hook (前向传播钩子) 来拦截捕获官方底层 MultiheadAttention 模块在推理时计算的交叉注意力权重，用于热力图可视化。
"""

import math
import torch
import torch.nn as nn
from .handwritten_transformer import PositionalEncoding

class TorchTransformer(nn.Module):
    """
    官方封装的 Transformer 模型 (PyTorch Native Seq2Seq Transformer)。
    基于 PyTorch 标准内置的 nn.TransformerEncoder 和 nn.TransformerDecoder 构建。
    与手写版 HandwrittenTransformer 保持相同输入输出接口和参数规模。
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=2, max_len=100, dropout=0.1):
        """
        初始化 PyTorch 官方封装的 Seq2Seq Transformer 网络。
        
        Args:
            src_vocab_size (int): 源序列词表大小。
            tgt_vocab_size (int): 目标序列词表大小。
            d_model (int): 模型嵌入和网络隐藏层维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏特征维度。
            num_layers (int): 编码器与解码器的堆叠层数。
            max_len (int): 允许的最大序列对齐长度。
            dropout (float): Dropout 比例。
        """
        super(TorchTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 1. 词嵌入映射层与位置编码层（与手写版共用相同的 PositionalEncoding 类，确保输入分布一致）
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 2. 官方编码器层与堆叠编码器
        # batch_first=True 使得张量形状支持标准的 [Batch, Seq_Len, d_model] 格式，无需繁琐的转置到 [Seq_Len, Batch, d_model]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 官方解码器层与堆叠解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. 线性输出层
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # 5. 用于存储钩子捕获的交叉注意力权重的占位变量
        self.last_cross_attn_weights = None
        self._register_attention_hook()
        
    def _register_attention_hook(self):
        """
        内部方法：通过动态替换（Monkey Patching）最后一层解码器中执行交叉注意力的 `_mha_block`，
        强制在底层调用 PyTorch 的 MultiheadAttention 时指定 `need_weights=True`，
        以此捕获被官方默认优化丢弃的注意力对齐矩阵，并保存到主模型的成员变量中。
        """
        if hasattr(self.decoder, 'layers') and len(self.decoder.layers) > 0:
            # 获取最后一层解码器
            last_decoder_layer = self.decoder.layers[-1]
            
            # 定义定制化的 _mha_block 方法
            def custom_mha_block(layer_self, x, mem, attn_mask, key_padding_mask, is_causal=False):
                # 显式传递 need_weights=True，以便获取注意力权重
                attn_out, attn_weights = layer_self.multihead_attn(
                    x,
                    mem,
                    mem,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    is_causal=is_causal,
                    need_weights=True
                )
                # 将捕获的注意力权重传回给主模型对象
                self.last_cross_attn_weights = attn_weights
                return layer_self.dropout2(attn_out)
            
            # 使用 lambda 包装方法并绑定到实例上
            # 当最后一层解码层调用 self._mha_block 时，将会路由到我们的 custom_mha_block 并传入实例对象
            last_decoder_layer._mha_block = lambda *args, **kwargs: custom_mha_block(last_decoder_layer, *args, **kwargs)

    def forward(self, src, tgt, src_pad_id=0, tgt_pad_id=0):
        """
        前向传播计算流程。
        
        Args:
            src (Tensor): 输入源序列，形状为 [B, L_src]
            tgt (Tensor): 目标序列，形状为 [B, L_tgt]
            src_pad_id (int): 源序列填充 ID，默认为 0。
            tgt_pad_id (int): 目标序列填充 ID，默认为 0。
            
        Returns:
            tuple: (模型词表对数输出 [B, L_tgt, tgt_vocab_size], 交叉注意力权重 [B, L_tgt, L_src])
        """
        # 1. 构造遮罩矩阵 (Masking)
        # 官方 nn.Transformer 期待的 key_padding_mask 形状为 [B, L]，类型为布尔型，
        # 其中 True 表示要遮蔽的 padding 位置，这与我们手写版的 make_src_mask (形状为 [B,1,1,L]，False为遮蔽) 有所不同。
        src_key_padding_mask = (src == src_pad_id)  # [B, L_src]
        tgt_key_padding_mask = (tgt == tgt_pad_id)  # [B, L_tgt]
        memory_key_padding_mask = src_key_padding_mask  # 编码器输出的 padding 情况同源序列 [B, L_src]
        
        # 构造解码器自注意力的因果遮罩 (Causal Mask)
        # 形状为 [L_tgt, L_tgt]，True 表示遮蔽，防止模型窥视未来数据
        tgt_len = tgt.size(1)
        device = tgt.device
        
        # 生成 PyTorch 官方的标准因果遮罩矩阵
        # 现代 PyTorch 中，generate_square_subsequent_mask 可以直接返回布尔掩码（True表示遮蔽，False表示可见）
        # 这里使用浮点掩码或布尔掩码均可，为方便起见，使用布尔掩码（与 torch.tril 取反一致）
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
        
        # 2. 词嵌入与位置编码
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 3. 官方编码器前向计算
        # nn.TransformerEncoder 参数包括 src_key_padding_mask
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # 4. 官方解码器前向计算
        # nn.TransformerDecoder 参数包括 tgt_mask (因果遮罩), tgt_key_padding_mask (目标填充遮罩), memory_key_padding_mask (源填充遮罩)
        dec_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 5. 线性分类输出
        output = self.generator(dec_out)
        
        # 返回输出和 Hook 捕获的交叉注意力权重
        # 注：在 inference/forward 结束后，钩子函数会自动把最后一层注意力权重写回 self.last_cross_attn_weights
        return output, self.last_cross_attn_weights


if __name__ == "__main__":
    # 本模块的简易测试与验证
    print("开始测试 torch_transformer.py 模块...")
    src_v = 40
    tgt_v = 40
    model = TorchTransformer(src_vocab_size=src_v, tgt_vocab_size=tgt_v, d_model=64, num_heads=2, d_ff=128, num_layers=1)
    
    # 模拟数据
    # Batch=2, src_len=8, tgt_len=6
    dummy_src = torch.tensor([[5, 12, 10, 4, 0, 0, 0, 0], [9, 8, 7, 6, 5, 4, 3, 2]], dtype=torch.long)
    dummy_tgt = torch.tensor([[1, 20, 22, 15, 2, 0], [1, 10, 11, 12, 13, 2]], dtype=torch.long)
    
    out, cross_attn = model(dummy_src, dummy_tgt)
    
    print(f"输出特征张量形状: {out.shape} (预期: [2, 6, {tgt_v}])")
    print(f"交叉注意力权重矩阵形状: {cross_attn.shape} (预期: [2, 6, 8])")
    
    assert out.shape == (2, 6, tgt_v), "输出形状异常！"
    assert cross_attn.shape == (2, 6, 8), "注意力权重矩阵形状异常！"
    print("官方 Transformer 模块前向推导测试成功！")
