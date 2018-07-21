# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_BiLSTM_lexicon.py
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


class BiLSTM_1(nn.Module):
    
    def __init__(self, args):
        super(BiLSTM_1, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        # self.embed = nn.Embedding(V, D, max_norm=config.max_norm)
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        # pretrained  embedding
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True, bias=True)
        # print(self.bilstm)
        self.hidden2label = nn.Linear(self.hidden_dim * 2 * 2, C, bias=True)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout(embed)
        x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(x)
        hidden = torch.cat(self.hidden, 0)
        hidden = torch.cat(hidden, 1)
        logit = self.hidden2label(F.tanh(hidden))
        return logit