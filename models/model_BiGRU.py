# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_BiGRU.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from DataUtils.Common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)

"""
Neural Networks model : Bidirection GRU
"""


class BiGRU(nn.Module):
    
    def __init__(self, args):
        super(BiGRU, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        # pretrained  embedding
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        # gru
        self.bigru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
        #  dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        embed = self.embed(input)
        embed = self.dropout(embed)
        input = embed.view(len(input), embed.size(1), -1)
        # gru
        gru_out, _ = self.bigru(input)
        gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        # gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = F.tanh(gru_out)
        # linear
        y = self.hidden2label(gru_out)
        logit = y
        return logit