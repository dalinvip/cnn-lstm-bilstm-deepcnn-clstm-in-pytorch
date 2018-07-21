# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CLSTM.py
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
    Neural Network: CLSTM
    Detail: The input first cross CNN model ,then the output of CNN as the input of LSTM
"""


class CLSTM(nn.Module):
    
    def __init__(self, args):
        super(CLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        # pretrained  embedding
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)

        # CNN
        KK = []
        for K in Ks:
            KK.append( K + 1 if K % 2 == 0 else K)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0)) for K in KK]
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K//2, 0)) for K in KK]
        # for cnn cuda
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()
        # LSTM
        self.lstm = nn.LSTM(D, self.hidden_dim, num_layers= self.num_layers, dropout=args.dropout)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)
        # dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        # CNN
        cnn_x = embed
        cnn_x = self.dropout(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 0)
        cnn_x = torch.transpose(cnn_x, 1, 2)
        # LSTM
        lstm_out, _ = self.lstm(cnn_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))
        # output
        logit = cnn_lstm_out

        return logit
