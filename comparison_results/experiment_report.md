# 模型对比实验报告

## 实验概述

本实验在相同的数据划分和训练配置下，对比了三种模型在情感分析任务上的表现：
1. **BiLSTM** - 双向LSTM基线模型
2. **Transformer** - 纯Transformer模型（使用Mean Pooling）
3. **CNN+Transformer** - CNN提取局部特征 + Transformer全局注意力融合

## 实验配置

| 参数 | 值 |
|------|-----|
| Epochs | 20 (带早停机制，patience=4) |
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
| **BiLSTM** | **83.14%** | **85.23%** | 1 | 114.50 | 4,736,852 |
| Transformer | 80.28% | 81.16% | 14 | 2647.41 | 6,364,652 |
| CNN+Transformer | 50.92% | 49.92% | 1 | 327.20 | 6,454,952 |

### 关键发现

1. **BiLSTM表现最佳**
   - 测试准确率达到85.23%，领先第二名约4%
   - 训练速度最快（114.50秒）
   - 参数量最少（约474万）
   - 在第1个epoch就达到最佳性能，收敛极快

2. **Transformer表现中等**
   - 测试准确率81.16%
   - 训练时间最长（约44分钟），是BiLSTM的23倍
   - 参数量较多（约636万）
   - 需要14个epoch才收敛到最佳性能

3. **CNN+Transformer表现不佳**
   - 测试准确率仅49.92%，接近随机猜测
   - 模型未能有效学习
   - 可能是CNN特征提取与Transformer融合方式需要优化

## 模型架构分析

### BiLSTM
```
Embedding(13838, 300) 
  -> BiLSTM(300, 150, bidirectional=True) 
  -> MaxPool1d 
  -> Linear(300, 150) 
  -> Linear(150, 2)
```
- 利用双向LSTM捕获前后文信息
- Max Pooling提取关键特征
- 结构简单高效

### Transformer
```
Embedding(13838, 300)
  -> PositionalEncoding
  -> TransformerEncoder(2 layers, 10 heads)
  -> Mean Pooling
  -> Linear(300, 150)
  -> Linear(150, 2)
```
- 使用位置编码保留序列信息
- 多头自注意力机制捕获全局依赖
- Mean Pooling聚合序列特征

### CNN+Transformer
```
Embedding(13838, 300)
  -> CNN(Kernels: 3,4,5, Channels: 100)
  -> MaxPool1d
  -> Linear(300, 300)
  -> PositionalEncoding
  -> TransformerEncoder(2 layers, 10 heads)
  -> Linear(300, 150)
  -> Linear(150, 2)
```
- CNN提取局部n-gram特征
- Transformer进行全局融合
- 但当前实现可能丢失了位置信息

## 结论与建议

### 结论

1. **对于小规模文本分类任务，BiLSTM仍是性价比最高的选择**
   - 参数量少，训练快，效果好
   - 在这个数据集上表现优于Transformer

2. **Transformer在这个任务上表现不如BiLSTM**
   - 可能是因为数据集规模较小（约7000样本）
   - Transformer通常需要大规模数据才能发挥优势
   - 训练时间成本较高

3. **CNN+Transformer的融合方式需要重新设计**
   - 当前实现未能有效工作
   - 建议尝试保留CNN输出的序列维度，让Transformer在序列级别进行注意力计算

### 改进建议

1. **对于Transformer模型**
   - 尝试不同的位置编码方式
   - 调整注意力头数和层数
   - 使用学习率预热策略

2. **对于CNN+Transformer模型**
   - 重新设计架构，保留CNN输出的序列维度
   - 考虑使用残差连接
   - 尝试不同的特征融合方式

3. **数据层面**
   - 尝试数据增强技术
   - 使用更大的预训练词向量
   - 尝试微调词向量

## 复现说明

运行以下命令复现实验：

```bash
python train_comparison.py \
  --epochs 20 \
  --batch_size 64 \
  --lr 0.001 \
  --dropout 0.3 \
  --patience 4 \
  --save_dir ./comparison_results
```

## 文件说明

- `model_Transformer.py` - 纯Transformer模型实现
- `model_CNN_Transformer.py` - CNN+Transformer模型实现
- `train_comparison.py` - 统一对比训练脚本
- `comparison_results.json` - 详细实验结果
