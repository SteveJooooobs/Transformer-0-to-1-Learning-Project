# -*- coding: utf-8 -*-
"""
实验 1：单头自注意力机制（Self-Attention）单步张量推导与可视化
本脚本不使用 PyTorch 高级 API，纯手动完成 Q、K、V 投影、注意力权重计算及加权求和，
并在终端以 ASCII 图表形式直观可视化注意力分布。
"""
import torch
import torch.nn.functional as F
import math

def run_experiment_1():
    print("=" * 60)
    print(" 实验 1: 纯张量单步推导与注意力可视化 ")
    print("=" * 60)
    
    # 1. 模拟输入数据
    # 假设输入句子为: ["我", "喜欢", "机器", "学习"]
    # 序列长度 seq_len = 4
    # 每个词的特征维度 d_model = 8 (为方便肉眼观察，设为较小的值)
    words = ["我", "喜欢", "机器", "学习"]
    seq_len = len(words)
    d_model = 8
    
    # 为了保证实验可重复性，设置随机种子
    torch.manual_seed(42)
    
    # 模拟 Embedding 层的输出，形状为 [batch_size=1, seq_len=4, d_model=8]
    X = torch.randn(1, seq_len, d_model)
    print(f"1. 输入张量 X 形状 (Batch, Seq_Len, d_model): {X.shape}")
    print("   X 的数值:\n", X.squeeze(0)) # 去掉 Batch 维度方便打印
    print("-" * 50)
    
    # 2. 手动创建权重矩阵 W_q, W_k, W_v
    # 投影维度为 d_k = d_v = 8
    d_k = d_model
    W_q = torch.randn(d_model, d_k)
    W_k = torch.randn(d_model, d_k)
    W_v = torch.randn(d_model, d_k)
    print(f"2. 初始化投影权重矩阵 W_q, W_k, W_v，形状皆为: {W_q.shape}")
    print("-" * 50)
    
    # 3. 计算 Q, K, V
    # 使用矩阵乘法，X 的 shape 是 [1, 4, 8]，W 的 shape 是 [8, 8]
    # 在 PyTorch 中，[1, 4, 8] @ [8, 8] 会自动对 Batch 维度进行广播，输出 [1, 4, 8]
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    
    print("3. 计算 Query(查询), Key(键), Value(值):")
    print(f"   Q 形状: {Q.shape}")
    print(f"   K 形状: {K.shape}")
    print(f"   V 形状: {V.shape}")
    print("-" * 50)
    
    # 4. 计算注意力原始得分 Scores = Q * K^T
    # 我们需要对 K 的最后两个维度进行转置才能做矩阵乘法。K.transpose(-2, -1) 形状为 [1, 8, 4]
    # Q [1, 4, 8] @ K^T [1, 8, 4] -> Scores [1, 4, 4]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print("4. 计算原始注意力得分 Scores (Q * K^T):")
    print(f"   Scores 形状 (Batch, Seq_Len, Seq_Len): {scores.shape}")
    print("   Scores 数值矩阵:\n", scores.squeeze(0))
    print("-" * 50)
    
    # 5. 缩放机制 (Scaling)
    # 为什么要除以 sqrt(d_k)？防止维度过大时点积数值过大，导致 Softmax 梯度消失
    scale = math.sqrt(d_k)
    scaled_scores = scores / scale
    print(f"5. 缩放得分 Scaled Scores (Scores / sqrt(d_k) [= {scale:.4f}]):")
    print("   Scaled Scores 数值矩阵:\n", scaled_scores.squeeze(0))
    print("-" * 50)
    
    # 6. 使用 Softmax 归一化，得到注意力权重 (Attention Weights)
    # 在最后一个维度 (dim=-1) 上做 Softmax，使每一行的权重和为 1
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print("6. 归一化注意力权重 Attention Weights (Softmax):")
    print(f"   Attention Weights 形状: {attention_weights.shape}")
    print("   权重数值矩阵:\n", attention_weights.squeeze(0))
    print("-" * 50)
    
    # 7. 对 Value 进行加权求和，得到最终输出 Output
    # Weights [1, 4, 4] @ V [1, 4, 8] -> Output [1, 4, 8]
    output = torch.matmul(attention_weights, V)
    print("7. 计算自注意力最终输出 Output (Weights * V):")
    print(f"   Output 形状: {output.shape}")
    print("   Output 数值矩阵:\n", output.squeeze(0))
    print("-" * 50)
    
    # 8. 终端 ASCII 字符可视化注意力图
    print("【注意力机制可视化热力图】(行代表当前词，列代表它对其他词的关注度)")
    print("注：百分比越大，说明该词对对应词的关注度越高。")
    print("=" * 60)
    
    # 打印表头
    header = f"{'单词':<6} | " + " | ".join([f"{w:^8}" for w in words])
    print(header)
    print("-" * (len(header) + 4))
    
    weights_np = attention_weights.squeeze(0).detach().numpy()
    for i, w_row in enumerate(words):
        row_str = f"{w_row:<6} | "
        cols = []
        for j, w_col in enumerate(words):
            val = weights_np[i, j]
            percent = f"{val * 100:6.2f}%"
            # 使用粗略字符标识强弱，> 40% 用 * 标识
            flag = "*" if val > 0.4 else " "
            cols.append(f"{percent}{flag}")
        row_str += " | ".join(cols)
        print(row_str)
    print("=" * 60)

if __name__ == "__main__":
    run_experiment_1()
