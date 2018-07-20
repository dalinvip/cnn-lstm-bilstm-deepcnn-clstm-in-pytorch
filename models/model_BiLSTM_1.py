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
            self.embed = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True)
        else:
            print("max_norm = {} |||||".format(args.max_norm))
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
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
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)
        print("self.hidden", self.hidden)

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.cuda is True:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)))

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        # x = x.view(len(x), x.size(1), -1)
        # x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, self.hidden = self.bilstm(x, self.hidden)
        # print(self.hidden)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)
        # bilstm_out = self.dropout(bilstm_out)

        # bilstm_out = self.hidden2label1(bilstm_out)
        # logit = self.hidden2label2(F.tanh(bilstm_out))

        logit = self.hidden2label(bilstm_out)

        return logit