import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
random.seed(1771)
torch.manual_seed(123)
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

        self.embed_no_static = nn.Embedding(V, D)
        self.embed_static = nn.Embedding(V_mui, D)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            pretrained_weight_static = np.array(args.pretrained_weight_static)
            self.embed_no_static.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embed_static.weight.data.copy_(torch.from_numpy(pretrained_weight_static))
            # whether to fixed the word embedding
            self.embed_no_static.weight.requires_grad = True
            # self.embed_static.weight.requires_grad = False

        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                # init.xavier_normal(conv.weight, gain=np.sqrt(args.init_weight_value))
                init.xavier_normal(conv.weight, gain=args.init_weight_value)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x):
        x_no_static = self.embed_no_static(x)
        # x_no_static = self.dropout(x_no_static)
        x_static = self.embed_static(x)
        x_static = Variable(x_static.data)
        # x_static = self.dropout(x_static)
        x = torch.stack([x_static, x_no_static], 1)
        # x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit