# -*- coding: utf-8 -*-
"""
实验 3：对比官方 PyTorch nn.MultiheadAttention 验证等价性
为了确保我们实验 2 自编写的多头自注意力机制数学上的正确性，
本脚本将我们的手写模型与 PyTorch 官方实现的 nn.MultiheadAttention 进行对比。
我们通过拷贝权重参数，输入相同的张量，来验证两者输出是否完全一致（误差在 1e-6 以内）。
"""
import torch
import torch.nn as nn
from demo_multi_head import MyMultiHeadAttention

def run_experiment_3():
    print("=" * 60)
    print(" 实验 3: 对比官方 nn.MultiheadAttention 验证等价性 ")
    print("=" * 60)
    
    # 1. 定义超参数与输入数据
    torch.manual_seed(100) # 固定种子以复现数据
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    
    X = torch.randn(batch_size, seq_len, d_model)
    print("输入 X 形状:", X.shape)
    
    # 2. 实例化两个模型
    # 手写的多头自注意力模型
    custom_mha = MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # 官方的多头自注意力层
    # batch_first=True 表示输入张量格式为 [batch, seq_len, d_model]（默认官方是 [seq_len, batch, d_model]）
    official_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
    
    # 3. 将手写模型的权重拷贝给官方模型
    # 官方模型为了追求极致效率，将 Q、K、V 的投影权重合并为了一个大矩阵 in_proj_weight (形状: [3*d_model, d_model])
    # 其排列顺序为: [Q_weight; K_weight; V_weight]
    with torch.no_grad():
        # 合并 Q, K, V 的权重矩阵与偏置
        qkv_weight = torch.cat([
            custom_mha.q_linear.weight,
            custom_mha.k_linear.weight,
            custom_mha.v_linear.weight
        ], dim=0)
        
        qkv_bias = torch.cat([
            custom_mha.q_linear.bias,
            custom_mha.k_linear.bias,
            custom_mha.v_linear.bias
        ], dim=0)
        
        # 拷贝给官方模型的投影参数
        official_mha.in_proj_weight.copy_(qkv_weight)
        official_mha.in_proj_bias.copy_(qkv_bias)
        
        # 拷贝最后的输出映射层权重和偏置
        official_mha.out_proj.weight.copy_(custom_mha.out_linear.weight)
        official_mha.out_proj.bias.copy_(custom_mha.out_linear.bias)
        
    print("权重拷贝完成！已同步手写模型和官方模型的网络参数。")
    print("-" * 50)
    
    # 4. 前向传播计算
    # 手写模型输出
    # 禁用梯度计算以方便观察
    custom_mha.eval()
    official_mha.eval()
    
    with torch.no_grad():
        out_custom, weights_custom = custom_mha(X)
        
        # 官方模型输出
        # 官方的 forward 需要传入 (query, key, value)
        # 在自注意力机制中，query = key = value = X
        # 返回值: (attn_output, attn_output_weights)
        out_official, weights_official = official_mha(X, X, X)
        
    print("-" * 50)
    print("5. 结果比对:")
    print(f"手写模型输出形状: {out_custom.shape}")
    print(f"官方模型输出形状: {out_official.shape}")
    
    # 6. 验证等价性 (使用 torch.allclose，判断在一定的浮点误差范围内是否相等)
    output_match = torch.allclose(out_custom, out_official, atol=1e-6)
    weights_match = torch.allclose(weights_custom.mean(dim=1), weights_official, atol=1e-6) 
    # 注: 官方返回的 weights_official 是各个头的平均注意力权重，所以用 weights_custom.mean(dim=1) 比对
    
    print("\n" + "=" * 60)
    print(f"【验证结论】:")
    print(f"  -> 手写与官方输出的特征特征是否一致: {'是 (SUCCESS)' if output_match else '否 (FAILED)'}")
    print(f"  -> 手写与官方注意权重均值是否一致: {'是 (SUCCESS)' if weights_match else '否 (FAILED)'}")
    
    if output_match and weights_match:
        print("  ==> 恭喜！手写的多头注意力层数学逻辑与 PyTorch 官方底层实现完全一致！")
    else:
        print("  ==> 警告：两者输出存在差异，请检查手写模型的维度转置与计算顺序！")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment_3()
