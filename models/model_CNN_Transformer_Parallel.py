# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CNN_Transformer_Parallel.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
import torch.nn.init as init
from DataUtils.Common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)

"""
Neural Networks model : CNN + Transformer (Parallel Architecture)
CNN captures local features, Transformer captures global dependencies
Features are fused in parallel and concatenated for classification
"""


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CNN_Transformer_Parallel(nn.Module):
    """
    并行结构：
    - 分支1: CNN提取局部特征 -> Max Pooling -> 固定维度向量
    - 分支2: Transformer提取全局特征 -> Mean Pooling -> 固定维度向量
    - 融合: 拼接两个分支的特征 -> 分类
    """
    
    def __init__(self, args):
        super(CNN_Transformer_Parallel, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        
        # Shared Embedding layer
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        
        # ========== Branch 1: CNN for local features ==========
        if args.wide_conv is True:
            print("using wide convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                     padding=(K//2, 0), dilation=1, bias=False) for K in Ks]
        else:
            print("using narrow convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
        print(self.convs1)
        
        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_normal_(conv.weight.data, gain=np.sqrt(args.init_weight_value))
        
        # for cnn cuda
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()
        
        # CNN output dimension
        self.cnn_output_dim = len(Ks) * Co
        
        # ========== Branch 2: Transformer for global features ==========
        self.pos_encoder = PositionalEncoding(D, dropout=args.dropout)
        
        self.d_model = D
        self.nhead = 10  # 300 must be divisible by nhead
        self.num_layers = 2
        self.dim_feedforward = D * 4
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=args.dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        
        print(self.transformer_encoder)
        
        # ========== Fusion and Classification ==========
        # Concatenate CNN and Transformer features
        self.fusion_dim = self.cnn_output_dim + D
        
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        
        # Classification layers
        self.hidden2label1 = nn.Linear(self.fusion_dim, self.fusion_dim // 2)
        self.hidden2label2 = nn.Linear(self.fusion_dim // 2, C)
        
    def forward(self, x):
        # x: (seq_len, batch_size) - batch_first=False format
        embed = self.embed(x)  # (seq_len, batch_size, D)
        embed = self.dropout_embed(embed)
        
        # ========== Branch 1: CNN Local Features ==========
        # Transpose to (batch_size, seq_len, D) for CNN
        embed_t = embed.transpose(0, 1)  # (batch_size, seq_len, D)
        cnn_input = embed_t.unsqueeze(1)  # (batch_size, 1, seq_len, D)
        
        # Apply convolutions and max pooling
        cnn_features = []
        for conv in self.convs1:
            conv_out = F.relu(conv(cnn_input)).squeeze(3)  # (batch_size, Co, new_seq_len)
            # Max pool to get (batch_size, Co)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            cnn_features.append(pooled)
        
        # Concatenate CNN features: (batch_size, len(Ks)*Co)
        cnn_out = torch.cat(cnn_features, 1)
        
        # ========== Branch 2: Transformer Global Features ==========
        # Apply positional encoding to embedded input
        embed_pos = self.pos_encoder(embed)  # (seq_len, batch_size, D)
        
        # Create padding mask
        padding_mask = (x.transpose(0, 1) == self.args.paddingId)  # (batch_size, seq_len)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(embed_pos, src_key_padding_mask=padding_mask)
        # transformer_out: (seq_len, batch_size, D)
        
        # Mean pooling (excluding padding positions)
        mask = (~padding_mask).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        transformer_out_t = transformer_out.transpose(0, 1)  # (batch_size, seq_len, D)
        sum_out = (transformer_out_t * mask).sum(dim=1)  # (batch_size, D)
        transformer_out = sum_out / mask.sum(dim=1).clamp(min=1)  # (batch_size, D)
        
        # ========== Feature Fusion ==========
        # Concatenate CNN and Transformer features
        fused_features = torch.cat([cnn_out, transformer_out], dim=1)  # (batch_size, fusion_dim)
        
        # Apply tanh activation
        fused_features = torch.tanh(fused_features)
        
        # Classification
        y = self.dropout(fused_features)
        y = self.hidden2label1(y)
        y = torch.tanh(y)
        y = self.hidden2label2(y)
        logit = y
        return logit
