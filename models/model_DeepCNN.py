# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_DeepCNN.py
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
    Neural Network: DEEP_CNN
    Detail: two layer cnn
"""


class DEEP_CNN(nn.Module):
    
    def __init__(self, args):
        super(DEEP_CNN, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        if args.max_norm is not None:
            print("max_norm = {} ".format(args.max_norm))
            self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True, padding_idx=args.paddingId)
        else:
            print("max_norm = {} ".format(args.max_norm))
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
            # fixed the word embedding
            self.embed.weight.requires_grad = True

        # cons layer
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K//2, 0), bias=True) for K in Ks]
        self.convs2 = [nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0), bias=True) for K in Ks]
        print(self.convs1)
        print(self.convs2)

        if args.init_weight:
            print("Initing W .......")
            for (conv1, conv2) in zip(self.convs1, self.convs2):
                init.xavier_normal(conv1.weight.data, gain=np.sqrt(args.init_weight_value))
                init.uniform(conv1.bias, 0, 0)
                init.xavier_normal(conv2.weight.data, gain=np.sqrt(args.init_weight_value))
                init.uniform(conv2.bias, 0, 0)

        # for cnn cuda
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        # for cnn cuda
        if self.args.cuda is True:
            for conv in self.convs2:
                conv = conv.cuda()

        # dropout
        self.dropout = nn.Dropout(args.dropout)
        # linear
        in_fea = len(Ks) * Co
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)


    def forward(self, x):
        one_layer = self.embed(x)  # (N,W,D) #  torch.Size([64, 43, 300])
        one_layer = one_layer.unsqueeze(1)  # (N,Ci,W,D)  #  torch.Size([64, 1, 43, 300])
        # one layer
        one_layer = [torch.transpose(F.relu(conv(one_layer)).squeeze(3), 1, 2) for conv in self.convs1] # torch.Size([64, 100, 36])
        # two layer
        two_layer = [F.relu(conv(one_layer.unsqueeze(1))).squeeze(3) for (conv, one_layer) in zip(self.convs2, one_layer)]
        # print("two_layer {}".format(two_layer[0].size()))
        # pooling
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in two_layer]   #  torch.Size([64, 100]) torch.Size([64, 100])
        output = torch.cat(output, 1)  # torch.Size([64, 300])
        # dropout
        output = self.dropout(output)
        # linear
        output = self.fc1(F.relu(output))
        logit = self.fc2(F.relu(output))
        return logit