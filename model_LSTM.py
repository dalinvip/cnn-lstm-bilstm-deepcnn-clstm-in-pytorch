import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.manual_seed(123)

class  LSTM(nn.Module):
    
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        # print(args)

        self.hidden_dim = args.lstm_hidden_dim
        # self.hidden_dim = 6400
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        # self.embed = nn.Embedding(V, D, max_norm=args.max_norm)
        self.embed = nn.Embedding(V, D)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # self.embed.weight.requires_grad = True
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=args.dropout)
        print(self.lstm)
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        self.hidden = self.init_hidden(args.batch_size)
        self.dropout = nn.Dropout(args.dropout)

    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        # return (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
        #          Variable(torch.zeros(1, batch_size, self.hidden_dim)))
        return (Variable(torch.randn(1, batch_size, self.hidden_dim)),
                 Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, x):
        embed = self.embed(x)
        # print("embed", embed.size())
        x = embed.view(len(x), embed.size(1), -1)
        # print("x.size()  {}   x.__class__  {}".format(x.size(), x.__class__))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # lstm_out = F.relu(lstm_out)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # lstm_out = F.tanh(lstm_out)
        # print("wwwwwwwww",lstm_out.size())
        # lstm_out = F.max_pool1d(lstm_out, kernel_size=1, stride=1)
        # lstm_out = F.max_pool1d(lstm_out, lstm_out.size(0))
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)
        # print("sssss", lstm_out.size())
        # lstm_out = self.dropout(lstm_out)
        # lstm_out = torch.transpose(lstm_out, 0, 1)
        # print("wwwwwwww", lstm_out.size())
        y = self.hidden2label(lstm_out)
        # logit = F.log_softmax(y)
        logit = y
        return logit