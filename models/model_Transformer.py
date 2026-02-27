# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_Transformer.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
from DataUtils.Common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)

"""
Neural Networks model : Pure Transformer with Mean Pooling
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


class Transformer(nn.Module):
    
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        
        # Embedding layer
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(D, dropout=args.dropout)
        
        # Transformer Encoder
        self.d_model = D
        self.nhead = 10  # Number of attention heads (300 must be divisible by nhead)
        self.num_layers = 2  # Number of transformer layers
        self.dim_feedforward = D * 4  # Feedforward dimension
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=args.dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        
        print(self.transformer_encoder)
        
        # Classification layers
        self.hidden2label1 = nn.Linear(self.d_model, self.d_model // 2)
        self.hidden2label2 = nn.Linear(self.d_model // 2, C)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x):
        # x: (seq_len, batch_size) - batch_first=False format
        embed = self.embed(x)  # (seq_len, batch_size, D)
        
        # Apply positional encoding
        embed = self.pos_encoder(embed)
        
        # Create padding mask
        # x: (seq_len, batch_size), transpose to (batch_size, seq_len) for mask
        padding_mask = (x.transpose(0, 1) == self.args.paddingId)  # (batch_size, seq_len)
        
        # Transformer encoding
        # output: (seq_len, batch_size, D)
        transformer_out = self.transformer_encoder(embed, src_key_padding_mask=padding_mask)
        
        # Transpose for pooling: (batch_size, seq_len, D)
        transformer_out = transformer_out.transpose(0, 1)
        
        # Mean pooling (excluding padding positions)
        # Expand mask to match transformer_out shape
        mask = (~padding_mask).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        sum_out = (transformer_out * mask).sum(dim=1)  # (batch_size, D)
        mean_out = sum_out / mask.sum(dim=1).clamp(min=1)  # (batch_size, D)
        
        # Apply tanh activation
        mean_out = torch.tanh(mean_out)
        
        # Classification
        y = self.dropout(mean_out)
        y = self.hidden2label1(y)
        y = torch.tanh(y)
        y = self.hidden2label2(y)
        logit = y
        return logit
