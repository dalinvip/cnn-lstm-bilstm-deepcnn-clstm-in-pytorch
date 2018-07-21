# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_DeepCNN_MUI.py
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
Description:
    the model is a mulit-channel DeepCNNS model, 
    the model use two external word embedding, and then,
    one of word embedding built from train/dev/test dataset,
    and it be used to no-fine-tune,other one built from only
    train dataset,and be used to fine-tune.

    my idea,even if the word embedding built from train/dev/test dataset, 
    whether can use fine-tune, in others words, whether can fine-tune with
    two external word embedding.
"""


class DEEP_CNN_MUI(nn.Module):
    
    def __init__(self, args):
        super(DEEP_CNN_MUI, self).__init__()
        self.args = args
        
        V = args.embed_num
        V_mui = args.embed_num_mui
        D = args.embed_dim
        C = args.class_num
        Ci = 2
        Co = args.kernel_num
        Ks = args.kernel_sizes
        if args.max_norm is not None:
            print("max_norm = {} ".format(args.max_norm))
            self.embed_no_static = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True, padding_idx=args.paddingId)
            self.embed_static = nn.Embedding(V_mui, D, max_norm=args.max_norm, scale_grad_by_freq=True, padding_idx=args.paddingId_mui)
        else:
            print("max_norm = {} ".format(args.max_norm))
            self.embed_no_static = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
            self.embed_static = nn.Embedding(V_mui, D, scale_grad_by_freq=True, padding_idx=args.paddingId_mui)
        if args.word_Embedding:
            self.embed_no_static.weight.data.copy_(args.pretrained_weight)
            self.embed_static.weight.data.copy_(args.pretrained_weight_static)
            # whether to fixed the word embedding
            self.embed_no_static.weight.requires_grad = False

        # cons layer
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K//2, 0), bias=True) for K in Ks]
        self.convs2 = [nn.Conv2d(1, Co, (K, D), stride=1, padding=(K//2, 0), bias=True) for K in Ks]
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
        x_no_static = self.embed_no_static(x)
        # x_no_static = self.dropout(x_no_static)
        x_static = self.embed_static(x)
        # fix the embedding
        x_static = Variable(x_static.data)
        # x_static = self.dropout(x_static)
        x = torch.stack([x_static, x_no_static], 1)
        one_layer = x  # (N,W,D) #  torch.Size([64, 43, 300])
        # print("one_layer {}".format(one_layer.size()))
        # one_layer = self.dropout(one_layer)
        # one_layer = one_layer.unsqueeze(1)  # (N,Ci,W,D)  #  torch.Size([64, 1, 43, 300])
        # one layer
        one_layer = [torch.transpose(F.relu(conv(one_layer)).squeeze(3), 1, 2).unsqueeze(1) for conv in self.convs1] # torch.Size([64, 100, 36])
        # one_layer = [F.relu(conv(one_layer)).squeeze(3).unsqueeze(1) for conv in self.convs1] # torch.Size([64, 100, 36])
        # print(one_layer[0].size())
        # print(one_layer[1].size())
        # two layer
        two_layer = [F.relu(conv(one_layer)).squeeze(3) for (conv, one_layer) in zip(self.convs2, one_layer)]
        # print("two_layer {}".format(two_layer[0].size()))
        # print("two_layer {}".format(two_layer[1].size()))
        # pooling
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in two_layer]   #  torch.Size([64, 100]) torch.Size([64, 100])
        output = torch.cat(output, 1)  # torch.Size([64, 300])
        # dropout
        output = self.dropout(output)
        # linear
        output = self.fc1(output)
        logit = self.fc2(F.relu(output))
        return logit