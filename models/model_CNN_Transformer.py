# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CNN_Transformer.py
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
Neural Networks model : CNN + Transformer
CNN captures local features, Transformer captures global dependencies with attention
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


class CNN_Transformer(nn.Module):
    
    def __init__(self, args):
        super(CNN_Transformer, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        
        # Embedding layer
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        
        # CNN layers for local feature extraction
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
        
        # Project CNN output to transformer dimension
        cnn_output_dim = len(Ks) * Co
        self.cnn_proj = nn.Linear(cnn_output_dim, D)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(D, dropout=args.dropout)
        
        # Transformer Encoder for global attention
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
        
        # Classification layers
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        self.hidden2label1 = nn.Linear(self.d_model, self.d_model // 2)
        self.hidden2label2 = nn.Linear(self.d_model // 2, C)
        
    def forward(self, x):
        # x: (seq_len, batch_size) - batch_first=False format
        embed = self.embed(x)  # (seq_len, batch_size, D)
        embed = self.dropout_embed(embed)
        
        # CNN feature extraction
        # Need to transpose to (batch_size, seq_len, D) for CNN
        embed_t = embed.transpose(0, 1)  # (batch_size, seq_len, D)
        # x: (batch_size, seq_len, D) -> (batch_size, 1, seq_len, D)
        cnn_input = embed_t.unsqueeze(1)
        
        # Apply convolutions and max pooling
        # Each conv output: (batch_size, Co, new_seq_len) after squeeze
        # Use max pooling to get fixed-size representation per filter
        cnn_features = []
        for conv in self.convs1:
            conv_out = F.relu(conv(cnn_input)).squeeze(3)  # (batch_size, Co, new_seq_len)
            # Max pool to get (batch_size, Co, 1) then squeeze to (batch_size, Co)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, Co)
            cnn_features.append(pooled)
        
        # Concatenate all CNN features: list of (batch_size, Co) -> (batch_size, len(Ks)*Co)
        cnn_out = torch.cat(cnn_features, 1)  # (batch_size, len(Ks)*Co)
        
        # Project to transformer dimension: (batch_size, D)
        # But we want to feed into transformer which expects sequence
        # So we add a sequence dimension: (batch_size, 1, D) -> (1, batch_size, D)
        cnn_out = self.cnn_proj(cnn_out)  # (batch_size, D)
        cnn_out = cnn_out.unsqueeze(0)  # (1, batch_size, D)
        
        # Apply positional encoding
        cnn_out = self.pos_encoder(cnn_out)
        
        # For single token sequence, no padding mask needed
        # Transformer encoding with global attention
        transformer_out = self.transformer_encoder(cnn_out)
        
        # Get output: (1, batch_size, D) -> (batch_size, D)
        transformer_out = transformer_out.squeeze(0)
        
        # Apply tanh activation
        mean_out = torch.tanh(transformer_out)
        
        # Classification
        y = self.dropout(mean_out)
        y = self.hidden2label1(y)
        y = torch.tanh(y)
        y = self.hidden2label2(y)
        logit = y
        return logit
