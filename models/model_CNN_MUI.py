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
class  CNN_MUI(nn.Module):
    
    def __init__(self, args):
        super(CNN_MUI,self).__init__()
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
            self.embed_no_static = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True)
            self.embed_static = nn.Embedding(V_mui, D, max_norm=args.max_norm, scale_grad_by_freq=True)
        else:
            print("max_norm = {} ".format(args.max_norm))
            self.embed_no_static = nn.Embedding(V, D, scale_grad_by_freq=True)
            self.embed_static = nn.Embedding(V_mui, D, scale_grad_by_freq=True)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            pretrained_weight_static = np.array(args.pretrained_weight_static)
            self.embed_no_static.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embed_static.weight.data.copy_(torch.from_numpy(pretrained_weight_static))
            # whether to fixed the word embedding
            self.embed_no_static.weight.requires_grad = True
            # self.embed_static.weight.requires_grad = False

        if args.wide_conv is True:
            print("using wide convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                     padding=(K//2, 0), bias=True) for K in Ks]
        else:
            print("using narrow convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
        # self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K // 2, 0)) for K in Ks]
        print(self.convs1)

        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_normal(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                init.uniform(conv.bias, 0, 0)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)

        in_fea = len(Ks) * Co
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)

        if args.batch_normalizations is True:
            print("using batch_normalizations in the model......")
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum,
                                            affine=args.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea//2, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x):
        x_no_static = self.embed_no_static(x)
        # x_no_static = self.dropout(x_no_static)
        x_static = self.embed_static(x)
        # fix the embedding
        x_static = Variable(x_static.data)
        # x_static = self.dropout(x_static)
        x = torch.stack([x_static, x_no_static], 1)
        # x = x.unsqueeze(1) # (N,Ci,W,D)
        x = self.dropout(x)
        if self.args.batch_normalizations is True:
            x = [F.relu(self.convs1_bn(conv(x))).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N,len(Ks)*Co)

        if self.args.batch_normalizations is True:
            x = self.fc1(x)
            logit = self.fc2(F.relu(x))
        else:
            x = self.fc1(x)
            logit = self.fc2(F.relu(x))
        return logit