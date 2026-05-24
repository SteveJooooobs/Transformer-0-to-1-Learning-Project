# 实验报告

> 自动生成的 Transformer 对比实验报告

---

## 实验结果汇总

| 实验名称 | d_model | num_heads | 使用PE | 参数量 | 最佳验证Loss | 最终验证PPL | 耗时(s) |
|---------|---------|-----------|--------|--------|-------------|------------|---------|
| Baseline（标准配置） | 128 | 4 | ✅ | 943,684 | 1.6482 | 5.20 | 95 |
| 无位置编码 | 128 | 4 | ❌ | 943,684 | 1.8896 | 6.62 | 94 |
| Head 数=1 | 128 | 1 | ✅ | 943,684 | 1.6800 | 5.37 | 88 |
| Head 数=2 | 128 | 2 | ✅ | 943,684 | 1.6620 | 5.27 | 95 |
| Head 数=4 | 128 | 4 | ✅ | 943,684 | 1.6434 | 5.17 | 97 |
| Head 数=8 | 128 | 8 | ✅ | 943,684 | 1.6535 | 5.23 | 99 |
| d_model=64 | 64 | 4 | ✅ | 242,500 | 1.7884 | 5.98 | 94 |
| d_model=128 | 128 | 4 | ✅ | 943,684 | 1.6493 | 5.20 | 94 |
| d_model=256 | 256 | 4 | ✅ | 3,722,308 | 1.5818 | 4.86 | 150 |

---

## 实验分析

### 1. Baseline vs 无位置编码

- Baseline 最佳验证 Loss: **1.6482**
- 无位置编码最佳验证 Loss: **1.8896**
- 差异: 0.2414

**结论**：位置编码对于 Transformer 的性能至关重要。去掉位置编码后，模型无法
区分不同位置的 token，导致序列建模能力显著下降。

### 2. 不同 Head 数的影响

- Head 数=1: 最佳验证 Loss = 1.6800
- Head 数=2: 最佳验证 Loss = 1.6620
- Head 数=4: 最佳验证 Loss = 1.6434
- Head 数=8: 最佳验证 Loss = 1.6535

**结论**：在 d_model=128 的设置下，4 个头通常是一个良好的平衡点。
过少的头（1个）可能限制模型的表达能力，而过多的头每个头的维度太小。

### 3. 不同 d_model 的影响

- d_model=64: 参数量=242,500, 最佳验证 Loss = 1.7884
- d_model=128: 参数量=943,684, 最佳验证 Loss = 1.6493
- d_model=256: 参数量=3,722,308, 最佳验证 Loss = 1.5818

**结论**：更大的 d_model 带来更多参数和更强的表达能力，但也需要更多的训练时间和数据。
对于 Tiny Shakespeare 这样的小数据集，d_model=128 是一个较好的选择。

---

## 生成文本样本

### Baseline（标准配置）

```
l I will be the provoked the prove the provost:
The words is the se thathathe thathathat wath t s s the sthe he the the the he he he he he he he he hathe he he hare he hare hare he wousthare wone hare
```

### 无位置编码

```
l the shall tand the the the the the she she she shalalll the the the the the the the the the the the the se the the the se the the the souse the the the the the the the se the the the the se the the 
```

### Head 数=1

```
l I have so my lord, and the see the see the soul be the see the heareateareat theat theare the t that the the the t the t that thene t the t the t thare the t the t the t the the the theathe the the 
```
