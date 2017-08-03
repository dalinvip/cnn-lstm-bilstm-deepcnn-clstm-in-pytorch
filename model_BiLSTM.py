import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.manual_seed(1233)
class  BiLSTM(nn.Module):
    
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        # self.embed = nn.Embedding(V, D, max_norm=args.max_norm)
        self.embed = nn.Embedding(V, D)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.bilstm = nn.LSTM(D, self.hidden_dim // 2, num_layers=1, dropout=args.dropout, bidirectional=True, bias=False)
        print(self.bilstm)

        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)
        self.hidden = self.init_hidden(args.batch_size)
        # self.dropout = nn.Dropout(args.dropout)

    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        # return (Variable(torch.zeros(2, batch_size, self.hidden_dim // 2)),
        #          Variable(torch.zeros(2, batch_size, self.hidden_dim // 2)))
        return (Variable(torch.randn(2, batch_size, self.hidden_dim // 2)),
                 Variable(torch.randn(2, batch_size, self.hidden_dim // 2)))

    def forward(self, x):
        embed = self.embed(x)
        x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, self.hidden = self.bilstm(x, self.hidden)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)
        logit = y
        return logit