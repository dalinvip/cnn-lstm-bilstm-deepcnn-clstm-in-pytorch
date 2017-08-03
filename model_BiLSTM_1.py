import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.manual_seed(1233)
class  BiLSTM_1(nn.Module):
    
    def __init__(self, args):
        super(BiLSTM_1, self).__init__()
        self.args = args
        # print(args)

        self.hidden_dim = args.lstm_hidden_dim
        # self.hidden_dim = 6400
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D, max_norm=args.max_norm)
        # self.embed = nn.Embedding(V, D)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # self.embed.weight.requires_grad = True
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=1, dropout=args.dropout, bidirectional=True, bias=True)
        print(self.bilstm)
        # self.hidden2label = nn.Linear(self.hidden_dim, C)
        self.hidden2label1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.hidden2label2 = nn.Linear(self.hidden_dim, C)
        self.hidden = self.init_hidden(args.batch_size)
        # self.dropout = nn.Dropout(args.dropout)

    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        # return (Variable(torch.zeros(2, batch_size, self.hidden_dim // 2)),
        #          Variable(torch.zeros(2, batch_size, self.hidden_dim // 2)))
        return (Variable(torch.randn(2, batch_size, self.hidden_dim)),
                 Variable(torch.randn(2, batch_size, self.hidden_dim)))

    def forward(self, x):
        embed = self.embed(x)
        # print("embed", embed.size())
        x = embed.view(len(x), embed.size(1), -1)
        # print("x.size()  {}   x.__class__  {}".format(x.size(), x.__class__))
        bilstm_out, self.hidden = self.bilstm(x, self.hidden)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        # bilstm_out = F.tanh(bilstm_out)
        # bilstm_out = F.max_pool1d(bilstm_out, kernel_size=1, stride=1)
        # bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        # bilstm_out = self.dropout(bilstm_out)
        # bilstm_out = F.tanh(bilstm_out)

        # y = self.hidden2label(bilstm_out[-1]).squeeze(1)
        y = self.hidden2label1(F.tanh(bilstm_out))
        # y = self.hidden2label1(bilstm_out[-1]).squeeze(1)
        y = self.hidden2label2(F.tanh(y))
        # logit = F.log_softmax(y)
        logit = y
        return logit