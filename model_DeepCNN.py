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
        # self.embed.weight.requires_grad = True
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # whether to fix the word embedding(True  = not, False = yes)
            # self.embed.weight.requires_grad = True

        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K//2, 0)) for K in Ks]
        self.convs2 = [nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0)) for K in Ks]

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
        # print("wwwwwww {} {} ", x.__class__, x.size())
        # print(x)

        # one_layer
        one_layer = self.embed(x) # (N,W,D) #  torch.Size([64, 43, 300])
        one_layer = one_layer.unsqueeze(1) # (N,Ci,W,D)  #  torch.Size([64, 1, 43, 300])
        # print("sss",one_layer.size())
        # print("qqqqq", x.size())
        # one_layer = [torch.transpose(F.relu(conv(one_layer)).squeeze(3), 1, 2) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        one_layer = [torch.transpose(F.relu(conv(one_layer)).squeeze(3), 1, 2) for conv in self.convs1] # torch.Size([64, 100, 36])
        # one_layer = [F.relu(conv(one_layer)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        # print("ddd", one_layer[0].size())   # torch.Size([64, 100, 36])
        # print(one_layer[0])
        # print(self.convs2[0])
        # print(torch.transpose(one_layer[0], 1, 2).unsqueeze(1).size())

        # two_layer = [F.relu(conv(torch.transpose(one_layer, 1, 2).unsqueeze(1))) for (conv, one_layer) in zip(self.convs2, one_layer)]
        two_layer = [F.relu(conv(one_layer.unsqueeze(1))).squeeze(3) for (conv, one_layer) in zip(self.convs2, one_layer)]
        # conv = self.convs2[0]
        # two_layer = [F.relu(conv(one_layer[0].unsqueeze(1)))]
        # print("rrrrrrrr {} ".format(two_layer[0].size(), two_layer[0].size()))

        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in two_layer]  # torch.Size([64, 100]) torch.Size([64, 100])
        # print("oursssss {} {}".format(output[0].size(), output[1].size()))
        output = torch.cat(output, 1)  # torch.Size([64, 300])
        # print("oursssss {}".format(output.size()))
        # one_layer = [torch.transpose(F.max_pool1d(i, i.size(2)).squeeze(2), 0, 1) for i in one_layer] #[(N,Co), ...]*len(Ks)
        # print("rrrr {} {} ".format(one_layer[0].size(),one_layer[1].size()))  # torch.Size([64, 100])
        # one_layer = torch.transpose(torch.cat(one_layer, 0), 0, 1)
        # print("tttttt", one_layer.size())  # torch.Size([64, 300])
        # one_layer = self.dropout(one_layer)
        #
        # print("wwwwwwww {} {} ", one_layer.__class__, one_layer.size())
        # # print(one_layer)
        #
        # # two_layer
        # two_layer = self.embed(one_layer)
        # two_layer = two_layer.unsqueeze(1)
        # # two_layer = one_layer.unsqueeze(1)
        # two_layer = [F.relu(conv(two_layer)).squeeze(3) for conv in self.convs2]
        # two_layer = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in two_layer]
        # two_layer = torch.cat(two_layer, 1)
        output = self.dropout(output)
        logit = self.fc1(output)
        return logit