import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.manual_seed(1235)
class  DEEP_CNN(nn.Module):
    
    def __init__(self, args):
        super(DEEP_CNN, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D)
        # word embedding
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # cons layer
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K//2, 0)) for K in Ks]
        self.convs2 = [nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0)) for K in Ks]
        # dropout
        self.dropout = nn.Dropout(args.dropout)
        # linear
        self.fc1 = nn.Linear(len(Ks)*Co, len(Ks)*Co // 2)
        self.fc2 = nn.Linear(len(Ks)*Co // 2, C)


    def forward(self, x):
        one_layer = self.embed(x) # (N,W,D) #  torch.Size([64, 43, 300])
        one_layer = one_layer.unsqueeze(1) # (N,Ci,W,D)  #  torch.Size([64, 1, 43, 300])
        # one layer
        one_layer = [torch.transpose(F.relu(conv(one_layer)).squeeze(3), 1, 2) for conv in self.convs1] # torch.Size([64, 100, 36])
        # two layer
        two_layer = [F.relu(conv(one_layer.unsqueeze(1))).squeeze(3) for (conv, one_layer) in zip(self.convs2, one_layer)]
        # pooling
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in two_layer]  # torch.Size([64, 100]) torch.Size([64, 100])
        output = torch.cat(output, 1)  # torch.Size([64, 300])
        # dropout
        output = self.dropout(output)
        # linear
        output = self.fc1(F.tanh(output))
        logit = self.fc2(F.tanh(output))
        return logit