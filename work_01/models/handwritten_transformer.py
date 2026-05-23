# -*- coding: utf-8 -*-
"""
手写 Transformer 模块：models/handwritten_transformer.py
本模块完全从零（不使用 PyTorch nn.Transformer 相关的高级封装）实现了经典的 Seq2Seq Transformer 架构。
包含以下组件：
1. PositionalEncoding：正弦/余弦位置编码，用于为字符输入注入位置信息。
2. MultiHeadAttention：支持掩码（Mask）的多头注意力层，用于计算自注意力和交叉注意力。
3. PositionwiseFeedForward：位置无关的前馈神经网络。
4. EncoderLayer & DecoderLayer：编码器层和解码器层，集成注意力、前馈网络、残差连接与层归一化（LayerNorm）。
5. Encoder & Decoder：堆叠多层实现深层编码与解码。
6. HandwrittenTransformer：将所有模块组装成完整的序列到序列（Seq2Seq）模型。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码层 (Positional Encoding)。
    为 Token Embedding 注入相对与绝对位置信息，因为 Transformer 架构本身不具备顺序感知能力。
    数学公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码矩阵并将其注册为 buffer 缓存。
        
        Args:
            d_model (int): 嵌入特征的维度。
            max_len (int): 允许的最大序列长度。
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建一个最大长度为 max_len，列为 d_model 维度的零张量
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置索引列 pos: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母的指数步长项 div_term: [d_model / 2]
        # 使用 exp(log(...)) 的形式计算可以保证数值计算稳定
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 对偶数索引列应用 sin 函数，奇数索引列应用 cos 函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 Batch 维度，形状变为 [1, max_len, d_model]，方便后续做广播相加
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer，表示它是模型状态的一部分但不需要计算梯度/进行参数更新
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播：将位置编码矩阵加到输入的 Word Embedding 张量中。
        
        Args:
            x (Tensor): 输入的字嵌入张量，形状为 [Batch_Size, Seq_Len, d_model]
            
        Returns:
            Tensor: 叠加了位置编码后的张量，形状与输入一致 [Batch_Size, Seq_Len, d_model]
        """
        # x.size(1) 为当前输入序列的长度，我们截取对应长度的位置编码相加
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力层 (Multi-Head Attention)。
    将特征维度划分为多个子头以捕获不同的上下文子空间信息。
    支持：
    1. 自注意力 (Self-Attention)
    2. 交叉注意力 (Cross-Attention / Encoder-Decoder Attention)
    3. 因果遮罩与 Padding 遮罩 (Masking)
    """
    
    def __init__(self, d_model, num_heads):
        """
        初始化线性投影权重并验证特征维度与头数的可整除性。
        
        Args:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头分配到的特征维度
        
        # 定义 Q, K, V 的线性映射层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # 输出的线性拼接层
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        """
        前向传播计算多头注意力。
        
        Args:
            q (Tensor): 查询张量，形状为 [B, L_q, d_model]
            k (Tensor): 键张量，形状为 [B, L_k, d_model]
            v (Tensor): 值张量，形状为 [B, L_v, d_model]，且 L_k 必须等于 L_v
            mask (Tensor, optional): 掩码矩阵，形状为 [B, 1, L_q, L_k] 或可广播的形状。
                                    其中 True 或 0 表示被遮蔽（需要被 fill 负无穷），False 或 1 表示正常参与计算。
                                    在本实现中，我们约定：传入的 mask 是布尔型掩码，其中 False 表示被遮蔽的部分。
                                    
        Returns:
            tuple: (注意力输出张量 [B, L_q, d_model], 注意力权重矩阵 [B, num_heads, L_q, L_k])
        """
        batch_size = q.size(0)
        
        # 1. 线性变换，投影得到 Q, K, V
        # 形状：[B, L, d_model]
        Q = self.q_linear(q)
        K = self.k_linear(k)
        V = self.v_linear(v)
        
        # 2. 拆分成多头，并将头维度移到前面
        # 变换过程：[B, L, d_model] -> [B, L, H, d_k] -> [B, H, L, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算点积得分并进行缩放运算 Scores = Q * K^T / sqrt(d_k)
        # Q 的形状为 [B, H, L_q, d_k]，K.transpose(-2, -1) 的形状为 [B, H, d_k, L_k]
        # 点积输出形状为 [B, H, L_q, L_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. 如果有掩码，将需要遮蔽的地方的得分设置为极大负数（如 -1e9），使 Softmax 后的权重趋近于 0
        if mask is not None:
            # 兼容布尔掩码与数值掩码，这里规定掩码中 False 或 0 对应的位置会被填充为负无穷
            scores = scores.masked_fill(mask == False, -1e9)
            
        # 5. Softmax 归一化，得到每个头上的注意力权重
        # 形状：[B, H, L_q, L_k]
        attn_weights = F.softmax(scores, dim=-1)
        
        # 6. 注意力权重乘以 Value 矩阵进行加权求和
        # [B, H, L_q, L_k] @ [B, H, L_k, d_k] -> [B, H, L_q, d_k]
        context = torch.matmul(attn_weights, V)
        
        # 7. 拼接 (Concatenate) 所有头的上下文，并变换回原始维度形状
        # [B, H, L_q, d_k] -> [B, L_q, H, d_k] -> [B, L_q, d_model]
        # 注：transpose 之后在内存中是不连续的，必须先调用 .contiguous() 才能进行 .view 操作
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # 8. 经过最后的线性映射层输出
        output = self.out_linear(context)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    位置无关的前馈神经网络 (Position-wise Feed-Forward Network)。
    对每个位置的特征向量独立进行两层线性映射，中间添加激活函数。
    结构：Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈网络的两层线性变换及 Dropout 比例。
        
        Args:
            d_model (int): 输入输出的特征维度。
            d_ff (int): 隐藏层中间特征的维度（通常是 d_model 的 4 倍）。
            dropout (float): Dropout 随机失活率。
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播：第一层映射 -> ReLU 激活 -> Dropout -> 第二层映射。
        
        Args:
            x (Tensor): 输入张量，形状为 [B, L, d_model]
            
        Returns:
            Tensor: 前馈输出张量，形状为 [B, L, d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    编码器层 (Encoder Layer)。
    每个层包含两个子层：
    1. 多头自注意力子层 (Multi-Head Self-Attention)
    2. 前馈网络子层 (Feed-Forward Network)
    每个子层周围都包含残差连接 (Residual Connection) 和层归一化 (Layer Normalization)。
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化编码器内部各组件。
        
        Args:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏层维度。
            dropout (float): 随机失活率。
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        # 随机失活
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播：Post-LN 结构（先计算子层，再做 Dropout，加残差，最后做 LayerNorm）。
        
        Args:
            x (Tensor): 输入特征张量，形状为 [B, L_src, d_model]
            mask (Tensor, optional): 源序列的 Padding 遮罩。
            
        Returns:
            Tensor: 编码器层输出，形状为 [B, L_src, d_model]
        """
        # 1. 自注意力子层 (Q=K=V=x)
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm_1(x + self.dropout_1(attn_out))
        
        # 2. 前馈网络子层
        ff_out = self.feed_forward(x)
        x = self.norm_2(x + self.dropout_2(ff_out))
        
        return x


class DecoderLayer(nn.Module):
    """
    解码器层 (Decoder Layer)。
    每个层包含三个子层：
    1. 带掩码的多头自注意力子层 (Masked Multi-Head Self-Attention)：确保解码时只能看到历史字符。
    2. 交叉注意力子层 (Encoder-Decoder Cross-Attention)：使解码器能够查询编码器的输出上下文。
    3. 前馈网络子层 (Feed-Forward Network)
    每个子层均通过残差连接和层归一化封装。
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化解码器层内部各组件。
        
        Args:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏层维度。
            dropout (float): 随机失活率。
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 定义三组 LayerNorm 层
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        # 定义三组 Dropout 层
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
    def forward(self, x, memory, self_mask=None, cross_mask=None):
        """
        前向传播：逐步计算自注意力、交叉注意力及前馈输出，并在每一步应用残差连接和归一化。
        
        Args:
            x (Tensor): 目标序列输入特征，形状为 [B, L_tgt, d_model]
            memory (Tensor): 编码器的最终隐藏状态输出 (Context)，形状为 [B, L_src, d_model]
            self_mask (Tensor, optional): 自注意力掩码（包含因果和 Padding 掩码）。
            cross_mask (Tensor, optional): 交叉注意力掩码（用于过滤源序列 Padding 字符）。
            
        Returns:
            tuple: (解码层输出 [B, L_tgt, d_model], 自注意力权重, 交叉注意力权重)
        """
        # 1. 掩码自注意力层 (Q=K=V=x)
        self_out, self_weights = self.self_attn(x, x, x, self_mask)
        x = self.norm_1(x + self.dropout_1(self_out))
        
        # 2. 交叉注意力层 (Q=x, K=V=memory)
        cross_out, cross_weights = self.cross_attn(x, memory, memory, cross_mask)
        x = self.norm_2(x + self.dropout_2(cross_out))
        
        # 3. 前馈网络层
        ff_out = self.feed_forward(x)
        x = self.norm_3(x + self.dropout_3(ff_out))
        
        return x, self_weights, cross_weights


class Encoder(nn.Module):
    """
    编码器 (Encoder)。
    由多层 EncoderLayer 堆叠而成，主要负责接收源序列并生成记忆上下文特征 (Memory)。
    """
    
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        """
        初始化堆叠编码层。
        
        Args:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏层维度。
            num_layers (int): 堆叠的编码层层数。
            dropout (float): 随机失活率。
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        前向传播：将输入特征依次喂入堆叠的每一层编码器中。
        
        Args:
            x (Tensor): 输入嵌入特征，形状为 [B, L_src, d_model]
            mask (Tensor, optional): 输入 Padding 遮罩。
            
        Returns:
            Tensor: 编码器最终上下文特征 (Memory)，形状为 [B, L_src, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    解码器 (Decoder)。
    由多层 DecoderLayer 堆叠而成，主要接收目标前缀和编码器的 Memory 输出并生成最终特征。
    """
    
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        """
        初始化堆叠解码层。
        
        Args:
            d_model (int): 特征维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络隐藏层维度。
            num_layers (int): 堆叠的解码层层数。
            dropout (float): 随机失活率。
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, memory, self_mask=None, cross_mask=None):
        """
        前向传播：遍历堆叠的每一层解码器，并收集最后一层解码层的注意力权重用于后续可视化。
        
        Args:
            x (Tensor): 目标前缀特征，形状为 [B, L_tgt, d_model]
            memory (Tensor): 编码器输出 Memory，形状为 [B, L_src, d_model]
            self_mask (Tensor, optional): 自注意力掩码。
            cross_mask (Tensor, optional): 交叉注意力掩码。
            
        Returns:
            tuple: (解码器最终特征 [B, L_tgt, d_model], 最后一层的自注意力权重, 最后一层的交叉注意力权重)
        """
        last_self_weights = None
        last_cross_weights = None
        
        for layer in self.layers:
            x, last_self_weights, last_cross_weights = layer(x, memory, self_mask, cross_mask)
            
        return self.norm(x), last_self_weights, last_cross_weights


class HandwrittenTransformer(nn.Module):
    """
    手写 Transformer 模型 (Handwritten Transformer Seq2Seq Model)。
    完整端到端架构，整合了嵌入层、位置编码、编码器、解码器及最后的线性映射分类头（Generator）。
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=2, max_len=100, dropout=0.1):
        """
        初始化完整的 Seq2Seq Transformer 网络。
        
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
        super(HandwrittenTransformer, self).__init__()
        
        # 1. 词嵌入映射层与位置编码层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 2. 核心编解码组件
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # 3. 线性分类输出层 (Generator)，将特征映射到词表概率分布
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        
    def make_src_mask(self, src, src_pad_id=0):
        """
        构建源序列 Padding 掩码。过滤被填充的 `<pad>` 字符。
        
        Args:
            src (Tensor): 输入的源序列 ID，形状为 [B, L_src]
            src_pad_id (int): 填充 Token 的 ID。
            
        Returns:
            Tensor: 布尔矩阵，形状为 [B, 1, 1, L_src]，为 True 表示保留计算，False 表示遮盖。
        """
        # (src != src_pad_id) 产生形状 [B, L_src] 的布尔张量
        # 插入维度，使其能够与 Attention Score 矩阵 [B, H, L_src, L_src] 自动广播对齐
        return (src != src_pad_id).unsqueeze(1).unsqueeze(2)
        
    def make_tgt_mask(self, tgt, tgt_pad_id=0):
        """
        构建目标序列复合掩码。
        它由以下两者联合逻辑与组成：
        1. 目标序列自身的 Padding 掩码：防止 Decoder 关注到 `<pad>`。
        2. 因果下三角遮罩 (Causal Look-Ahead Mask)：防止当前时间步窥视未来的 Token。
        
        Args:
            tgt (Tensor): 目标序列输入 ID，形状为 [B, L_tgt]
            tgt_pad_id (int): 目标序列填充 ID。
            
        Returns:
            Tensor: 目标掩码矩阵，形状为 [B, 1, L_tgt, L_tgt] 且为布尔类型。
        """
        tgt_len = tgt.size(1)
        
        # 1. 目标序列 Padding 掩码，形状为 [B, 1, 1, L_tgt] 扩展为 [B, 1, 1, L_tgt]
        tgt_pad_mask = (tgt != tgt_pad_id).unsqueeze(1).unsqueeze(2)
        
        # 2. 因果下三角掩码，利用 torch.tril 构建下三角矩阵（1 表示正常计算，0 表示遮罩）
        # 形状为 [L_tgt, L_tgt]
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        # 扩展维度为 [1, 1, L_tgt, L_tgt]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        
        # 3. 两者做逻辑与 (AND) 运算，得到最终的解码器自注意力掩码
        tgt_mask = tgt_pad_mask & causal_mask
        return tgt_mask

    def forward(self, src, tgt, src_pad_id=0, tgt_pad_id=0):
        """
        前向传播计算流程。
        
        Args:
            src (Tensor): 输入源序列，形状为 [B, L_src]
            tgt (Tensor): 目标序列（注意：训练时此处传入已经向右错位、不含最末 Token 的目标序列），形状为 [B, L_tgt]
            src_pad_id (int): 源序列填充 ID，默认为 0。
            tgt_pad_id (int): 目标序列填充 ID，默认为 0。
            
        Returns:
            tuple: (模型词表对数输出 [B, L_tgt, tgt_vocab_size], 交叉注意力权重 [B, num_heads, L_tgt, L_src])
        """
        # 1. 构造掩码
        # src_mask 形状：[B, 1, 1, L_src]
        src_mask = self.make_src_mask(src, src_pad_id)
        # tgt_mask 形状：[B, 1, L_tgt, L_tgt]
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_id)
        # cross_mask 形状与 src_mask 完全一致：[B, 1, 1, L_src]，使 Decoder 避免对 src 的 pad 算注意力
        cross_mask = src_mask
        
        # 2. 词嵌入映射与位置编码
        # 注意：加位置编码前，需要乘上 sqrt(d_model) 来对 Word Embedding 的值进行缩放
        # 这是为了在特征维度较大时，防止位置编码的值喧宾夺主
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 3. 编码器计算
        memory = self.encoder(src_emb, src_mask)
        
        # 4. 解码器计算
        dec_out, _, cross_weights = self.decoder(tgt_emb, memory, tgt_mask, cross_mask)
        
        # 5. 线性分类映射，输出对数概率分布
        output = self.generator(dec_out)
        
        return output, cross_weights


if __name__ == "__main__":
    # 本模块的简单形状与前向验证
    print("开始测试 handwritten_transformer.py 模块...")
    src_v = 40
    tgt_v = 40
    model = HandwrittenTransformer(src_vocab_size=src_v, tgt_vocab_size=tgt_v, d_model=64, num_heads=2, d_ff=128, num_layers=1)
    
    # 模拟数据
    # Batch=2, src_len=8, tgt_len=6
    dummy_src = torch.tensor([[5, 12, 10, 4, 0, 0, 0, 0], [9, 8, 7, 6, 5, 4, 3, 2]], dtype=torch.long)
    dummy_tgt = torch.tensor([[1, 20, 22, 15, 2, 0], [1, 10, 11, 12, 13, 2]], dtype=torch.long)
    
    out, cross_attn = model(dummy_src, dummy_tgt)
    
    print(f"输出特征张量形状: {out.shape} (预期: [2, 6, {tgt_v}])")
    print(f"注意力权重矩阵形状: {cross_attn.shape} (预期: [2, 2, 6, 8])")
    
    assert out.shape == (2, 6, tgt_v), "输出形状异常！"
    assert cross_attn.shape == (2, 2, 6, 8), "注意力权重矩阵形状异常！"
    print("手写 Transformer 模块前向推导测试成功！")
