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
        self.embed = nn.Embedding(V, D)
        # pretrained  embedding
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # gru
        self.bigru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
        # hidden
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)
        # dropout
        self.dropout = nn.Dropout(args.dropout)

    def init_hidden(self, num_layers, batch_size):
        if self.args.cuda is True:
            return Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim))

    def forward(self, input):
        embed = self.embed(input)
        embed = self.dropout(embed)
        input = embed.view(len(input), embed.size(1), -1)
        # gru
        gru_out, hidden = self.bigru(input, self.hidden)
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