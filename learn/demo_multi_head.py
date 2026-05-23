# -*- coding: utf-8 -*-
"""
实验 2：从零手写多头自注意力类（Multi-Head Attention）
本脚本使用 PyTorch 的 nn.Module 搭建一个白盒化的多头自注意力层。
通过输入真实大小的张量，打印每一步维度变换的生命周期，展示数据是如何被“拆分、多头计算、合并”的。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MyMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除！"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头分配到的维度
        
        # 定义 Q, K, V 的线性投影层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # 最后的输出投影层
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        print(f"\n[Forward] 1. 输入张量形状 x: {x.shape} (Batch={batch_size}, Seq_Len={seq_len}, d_model={self.d_model})")
        
        # 1. 线性投影得到 Q, K, V
        # 形状依然是: [batch_size, seq_len, d_model]
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        print(f"[Forward] 2. 线性投影输出 Q/K/V 形状: {Q.shape}")
        
        # 2. 拆分成多头 (Split into multi-heads)
        # 将最终的 d_model 拆为 num_heads * d_k
        # 维度变换: [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        print(f"[Forward] 3. 维度拆分后 Q/K/V 形状: {Q.shape} (拆为 {self.num_heads} 个头，每头 {self.d_k} 维)")
        
        # 3. 转置 (Transpose) 以便进行批矩阵乘法 (Batch Matrix Multiplication)
        # 为了让 num_heads 处于计算的批次维度，需要交换维度 1 和 2
        # 变换后形状: [batch_size, num_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        print(f"[Forward] 4. 转置交换后 Q/K/V 形状: {Q.shape} (满足并行计算格式)")
        
        # 4. 计算注意力得分 scores = (Q * K^T) / sqrt(d_k)
        # Q 形状: [B, H, L, d_k], K^T 形状: [B, H, d_k, L] -> scores 形状: [B, H, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        print(f"[Forward] 5. 计算缩放点积得分 scores 形状: {scores.shape} (每个头拥有独立的 L x L 注意力矩阵)")
        
        # 5. Softmax 归一化得到注意力权重
        weights = F.softmax(scores, dim=-1)
        print(f"[Forward] 6. 归一化注意力权重 weights 形状: {weights.shape}")
        
        # 6. 注意力权重乘以 V
        # weights [B, H, L, L] @ V [B, H, L, d_k] -> context [B, H, L, d_k]
        context = torch.matmul(weights, V)
        print(f"[Forward] 7. 加权求和上下文 context 形状: {context.shape}")
        
        # 7. 合并多头 (Concatenate)
        # 先将维度转置回: [batch_size, seq_len, num_heads, d_k]
        # 注意: transpose 之后张量在内存中是不连续的，必须调用 contiguous() 才能使用 view 重新塑形
        context = context.transpose(1, 2).contiguous()
        # 将最后两个维度合并: [batch_size, seq_len, num_heads * d_k] -> [batch_size, seq_len, d_model]
        context = context.view(batch_size, seq_len, self.d_model)
        print(f"[Forward] 8. 拼接多头并还原维度后 context 形状: {context.shape}")
        
        # 8. 通过最后的输出投影层
        output = self.out_linear(context)
        print(f"[Forward] 9. 经过最后一层线性映射后 output 形状: {output.shape}")
        
        return output, weights

def run_experiment_2():
    print("=" * 60)
    print(" 实验 2: 从零手写多头自注意力模型 ")
    print("=" * 60)
    
    # 模拟输入参数
    batch_size = 2
    seq_len = 5
    d_model = 12   # embedding 维度
    num_heads = 3  # 3 个头，则 d_k = 12 / 3 = 4
    
    # 创建模拟输入，表示 2 句话，每句话 5 个词，每个词 12 维向量
    X = torch.randn(batch_size, seq_len, d_model)
    
    # 实例化我们的多头自注意力模型
    mha = MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # 运行前向传播
    output, weights = mha(X)
    
    print("\n" + "=" * 60)
    print(" 维度变换生命周期一览 ")
    print("=" * 60)
    print(f" 输入 X:          {X.shape}")
    print(f" 拆分头:          [Batch, Seq_Len, Num_Heads, d_k] -> [2, 5, 3, 4]")
    print(f" 转置准备计算:    [Batch, Num_Heads, Seq_Len, d_k] -> [2, 3, 5, 4]")
    print(f" 注意力权重 weights: {weights.shape}")
    print(f" 输出 Output:     {output.shape}")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment_2()
