# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CNN_BiGRU.py
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
    Neural Network: CNN_BiGRU
    Detail: the input crosss cnn model and GRU model independly, then the result of both concat
"""


class CNN_BiGRU(nn.Module):
    
    def __init__(self, args):
        super(CNN_BiGRU,self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.C = C
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        # pretrained  embedding
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)

        # CNN
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D), padding=(K//2, 0), stride=1) for K in Ks]
        print(self.convs1)
        # for cnn cuda
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        # BiGRU
        self.bigru = nn.GRU(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True, bias=True)

        # linear
        L = len(Ks) * Co + self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

        # dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout(embed)
        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)
        # BiGRU
        bigru_x = embed.view(len(x), embed.size(1), -1)
        bigru_x, _ = self.bigru(bigru_x)
        bigru_x = torch.transpose(bigru_x, 0, 1)
        bigru_x = torch.transpose(bigru_x, 1, 2)
        # bilstm_out = F.tanh(bilstm_out)
        bigru_x = F.max_pool1d(bigru_x, bigru_x.size(2)).squeeze(2)
        bigru_x = F.tanh(bigru_x)

        # CNN and BiGRU CAT
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bigru_x = torch.transpose(bigru_x, 0, 1)
        cnn_bigru_out = torch.cat((cnn_x, bigru_x), 0)
        cnn_bigru_out = torch.transpose(cnn_bigru_out, 0, 1)

        # linear
        cnn_bigru_out = self.hidden2label1(F.tanh(cnn_bigru_out))
        logit = self.hidden2label2(F.tanh(cnn_bigru_out))

        return logit