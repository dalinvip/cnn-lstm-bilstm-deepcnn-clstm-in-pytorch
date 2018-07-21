# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_HighWay_BiLSTM_1.py
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
    Neural Networks model : Highway Networks and BiLSTM
    Highway Networks : git@github.com:bamtercelboo/pytorch_Highway_Networks.git
"""


class HighWay_BiLSTM_1(nn.Module):
    def __init__(self, args):
        super(HighWay_BiLSTM_1, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.dropout = nn.Dropout(args.dropout)
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
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

            # init weight of lstm gate
            self.bilstm.all_weights[0][3].data[20:40].fill_(1)
            self.bilstm.all_weights[0][3].data[0:20].fill_(0)
            self.bilstm.all_weights[0][3].data[40:80].fill_(0)
            # self.bilstm.all_weights[0][3].data[40:].fill_(0)
            self.bilstm.all_weights[0][2].data[20:40].fill_(1)
            self.bilstm.all_weights[0][2].data[0:20].fill_(0)
            self.bilstm.all_weights[0][2].data[40:80].fill_(0)
            # self.bilstm.all_weights[0][2].data[40:].fill_(0)
            self.bilstm.all_weights[1][3].data[20:40].fill_(1)
            self.bilstm.all_weights[1][3].data[0:20].fill_(0)
            self.bilstm.all_weights[1][3].data[40:80].fill_(0)
            # self.bilstm.all_weights[1][3].data[40:].fill_(0)
            self.bilstm.all_weights[1][2].data[20:40].fill_(1)
            self.bilstm.all_weights[1][2].data[0:20].fill_(0)
            self.bilstm.all_weights[1][2].data[40:80].fill_(0)
            # self.bilstm.all_weights[1][2].data[40:].fill_(0)

        self.hidden2label1 = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2, bias=True)

        # highway gate layer
        self.gate_layer = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2, bias=True)

        # last liner
        self.logit_layer = nn.Linear(in_features=self.hidden_dim * 2, out_features=C, bias=True)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        bilstm_out, _ = self.bilstm(x)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2))
        bilstm_out = bilstm_out.squeeze(2)

        hidden2lable = self.hidden2label1(F.tanh(bilstm_out))

        gate_layer = F.sigmoid(self.gate_layer(bilstm_out))
        # calculate highway layer values
        gate_hidden_layer = torch.mul(hidden2lable, gate_layer)
        # if write like follow ,can run,but not equal the HighWay NetWorks formula
        # gate_input = torch.mul((1 - gate_layer), hidden2lable)
        gate_input = torch.mul((1 - gate_layer), bilstm_out)
        highway_output = torch.add(gate_hidden_layer, gate_input)

        logit = self.logit_layer(highway_output)

        return logit