# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_HighWay_CNN.py
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
    Neural Networks model : Highway Networks and CNN
    Highway Networks : git@github.com:bamtercelboo/pytorch_Highway_Networks.git
"""


class HighWay_CNN(nn.Module):
    
    def __init__(self, args):
        super(HighWay_CNN, self).__init__()
        self.args = args
        
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

        print("dddd {} ".format(self.embed.weight.data.size()))

        if args.wide_conv is True:
            print("using wide convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                     padding=(K//2, 0), dilation=1, bias=True) for K in Ks]
        else:
            print("using narrow convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
        print(self.convs1)

        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_normal(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                fan_in, fan_out = HighWay_CNN.calculate_fan_in_and_fan_out(conv.weight.data)
                print(" in {} out {} ".format(fan_in, fan_out))
                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
                init.uniform(conv.bias, 0, 0)

        # for cnn cuda
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        self.dropout = nn.Dropout(args.dropout)

        in_fea = len(Ks) * Co
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea, bias=True)

        # highway gate layer
        self.gate_layer = nn.Linear(in_features=in_fea, out_features=in_fea, bias=True)

        # last liner
        self.logit_layer = nn.Linear(in_features=in_fea, out_features=C, bias=True)

        # whether to use batch normalizations
        if args.batch_normalizations is True:
            print("using batch_normalizations in the model......")
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum,
                                            affine=args.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea//2, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)

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
        x = self.embed(x)  # (N,W,D)
        x = self.dropout(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        if self.args.batch_normalizations is True:
            x = [self.convs1_bn(F.tanh(conv(x))).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        if self.args.batch_normalizations is True:
            x = self.fc1_bn(self.fc1(x))
            fc = self.fc2_bn(self.fc2(F.tanh(x)))
        else:
            fc = self.fc1(x)

        gate_layer = F.sigmoid(self.gate_layer(x))

        # calculate highway layer values
        gate_fc_layer = torch.mul(fc, gate_layer)
        # if write like follow ,can run,but not equal the HighWay NetWorks formula
        # gate_input = torch.mul((1 - gate_layer), fc)
        gate_input = torch.mul((1 - gate_layer), x)
        highway_output = torch.add(gate_fc_layer, gate_input)

        logit = self.logit_layer(highway_output)

        return logit