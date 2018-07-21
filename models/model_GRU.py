# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_GRU.py
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
Neural Networks model : GRU
"""


class GRU(nn.Module):
    
    def __init__(self, args):
        super(GRU, self).__init__()
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
        self.gru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        #  dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        embed = self.embed(input)
        input = embed.view(len(input), embed.size(1), -1)
        lstm_out, _ = self.gru(input)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)
        # linear
        y = self.hidden2label(lstm_out)
        logit = y
        return logit