# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_BiLSTM_1.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
from DataUtils.Common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)

"""
Neural Networks model : Bidirection LSTM
"""


class BiLSTM_1(nn.Module):
    def __init__(self, args):
        super(BiLSTM_1, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        if args.max_norm is not None:
            print("max_norm = {} ".format(args.max_norm))
            self.embed = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True, padding_idx=args.paddingId)
            # pretrained  embedding
            if args.word_Embedding:
                self.embed.weight.data.copy_(args.pretrained_weight)
        else:
            print("max_norm = {} |||||".format(args.max_norm))
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
            # pretrained  embedding
            if args.word_Embedding:
                self.embed.weight.data.copy_(args.pretrained_weight)

        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, bias=True, bidirectional=True,
                              dropout=self.args.dropout)
        print(self.bilstm)
        if args.init_weight:
            print("Initing W .......")
            init.xavier_normal(self.bilstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[1][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[1][1], gain=np.sqrt(args.init_weight_value))

        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        bilstm_out, _ = self.bilstm(x)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)

        logit = self.hidden2label(bilstm_out)

        return logit