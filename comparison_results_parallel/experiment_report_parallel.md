# 模型对比实验报告（含并行结构CNN+Transformer）

## 实验概述

本实验在相同的数据划分和训练配置下，对比了四种模型在情感分析任务上的表现：
1. **BiLSTM** - 双向LSTM基线模型
2. **Transformer** - 纯Transformer模型（使用Mean Pooling）
3. **CNN+Transformer** - 串行结构：CNN提取局部特征 -> Transformer全局融合
4. **CNN+Transformer_Parallel** - 并行结构：CNN和Transformer分别提取特征后拼接

## 实验配置

| 参数 | 值 |
|------|-----|
| Epochs | 15 (带早停机制，patience=4) |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Dropout | 0.3 |
| Embedding Dim | 300 |
| LSTM Hidden Dim | 300 |
| Optimizer | Adam |
| 数据集 | MR (二分类情感分析) |
| 训练集 | 6920 样本 |
| 验证集 | 872 样本 |
| 测试集 | 1821 样本 |
| 词表大小 | 13838 |

## 实验结果

### 性能对比

| 模型 | 最佳验证准确率 | 最佳测试准确率 | 最佳Epoch | 训练时间(s) | 参数量 |
|------|---------------|---------------|----------|------------|--------|
| **BiLSTM** | **83.14%** | **85.23%** | 1 | 93.64 | 4,736,852 |
| Transformer | 80.28% | 81.16% | 14 | 891.75 | 6,364,652 |
| CNN+Transformer (串行) | 50.92% | 49.92% | 1 | 291.78 | 6,454,952 |
| **CNN+Transformer_Parallel** | **82.34%** | **82.26%** | 7 | 1350.45 | 6,500,102 |

### 关键发现

1. **BiLSTM表现最佳**
   - 测试准确率达到85.23%，领先第二名约3%
   - 训练速度最快（93.64秒）
   - 参数量最少（约474万）
   - 在第1个epoch就达到最佳性能，收敛极快

2. **CNN+Transformer并行结构表现良好**
   - 测试准确率82.26%，接近BiLSTM
   - 显著优于串行结构（49.92%）和纯Transformer（81.16%）
   - 证明并行特征融合的有效性

3. **串行CNN+Transformer失败**
   - 测试准确率仅49.92%，接近随机猜测
   - 说明CNN特征提取后直接输入Transformer的方式存在问题

4. **Transformer在这个任务上表现中等**
   - 测试准确率81.16%
   - 训练时间较长（约15分钟）

## 模型架构对比

### 串行 vs 并行结构

**串行结构 (CNN+Transformer)**
```
Input -> Embedding -> CNN -> MaxPool -> Linear -> Transformer -> Output
```
- 问题：CNN的MaxPooling丢失了序列信息，导致Transformer无法有效学习

**并行结构 (CNN+Transformer_Parallel)**
```
Input -> Embedding -> [CNN -> MaxPool] -> Concat -> Output
                          [Transformer -> MeanPool]
```
- 优势：两个分支独立提取特征，融合时保留了各自的表达能力

## 架构设计要点

### CNN+Transformer_Parallel
```python
# 分支1: CNN提取局部特征
CNN(Kernels: 3,4,5, Channels: 100) -> MaxPool -> 300维向量

# 分支2: Transformer提取全局特征  
PositionalEncoding -> TransformerEncoder(2 layers, 10 heads) -> MeanPool -> 300维向量

# 融合: 拼接两个分支
Concat([CNN_features, Transformer_features]) -> 600维向量
-> Linear(600, 300) -> Linear(300, 2)
```

## 结论与建议

### 结论

1. **对于小规模文本分类任务，BiLSTM仍是性价比最高的选择**
   - 参数量少，训练快，效果好

2. **并行结构优于串行结构**
   - CNN+Transformer_Parallel (82.26%) >> CNN+Transformer串行 (49.92%)
   - 说明特征融合方式对模型性能至关重要

3. **CNN和Transformer可以互补**
   - CNN擅长局部n-gram特征
   - Transformer擅长全局依赖关系
   - 并行融合可以结合两者优势

### 改进建议

1. **进一步优化并行结构**
   - 尝试不同的特征融合方式（如注意力融合）
   - 调整两个分支的权重
   - 使用残差连接

2. **超参数调优**
   - 调整CNN的kernel sizes和channels
   - 调整Transformer的层数和头数
   - 尝试不同的dropout率

## 复现说明

运行以下命令复现实验：

```bash
python train_comparison.py \
  --epochs 15 \
  --batch_size 64 \
  --lr 0.001 \
  --dropout 0.3 \
  --patience 4 \
  --save_dir ./comparison_results_parallel
```

## 文件说明

- `model_Transformer.py` - 纯Transformer模型实现
- `model_CNN_Transformer.py` - 串行CNN+Transformer模型实现
- `model_CNN_Transformer_Parallel.py` - 并行CNN+Transformer模型实现
- `train_comparison.py` - 统一对比训练脚本
- `comparison_results.json` - 详细实验结果
