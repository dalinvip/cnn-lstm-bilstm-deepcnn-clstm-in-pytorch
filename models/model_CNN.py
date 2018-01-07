import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)

"""
    Neural Network: CNN
"""


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        if args.max_norm is not None:
            print("max_norm = {} ".format(args.max_norm))
            self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True)
            # self.embed.weight.data.uniform(-0.1, 0.1)
        else:
            print("max_norm = {} ".format(args.max_norm))
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # fixed the word embedding
            self.embed.weight.requires_grad = True
        print("dddd {} ".format(self.embed.weight.data.size()))

        if args.wide_conv is True:
            print("using wide convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                     padding=(K//2, 0), dilation=1, bias=False) for K in Ks]
        else:
            print("using narrow convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
        print(self.convs1)

        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_normal(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)
                print(" in {} out {} ".format(fan_in, fan_out))
                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
        # for cnn cuda
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        in_fea = len(Ks) * Co
        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)
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
        x = self.dropout_embed(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        if self.args.batch_normalizations is True:
            x = [self.convs1_bn(F.tanh(conv(x))).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        if self.args.batch_normalizations is True:
            x = self.fc1_bn(self.fc1(x))
            logit = self.fc2_bn(self.fc2(F.tanh(x)))
        else:
            logit = self.fc(x)
        return logit
