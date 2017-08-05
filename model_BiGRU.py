import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.manual_seed(123)

class  BiGRU(nn.Module):
    
    def __init__(self, args):
        super(BiGRU, self).__init__()
        self.args = args
        # print(args)

        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        # self.embed = nn.Embedding(V, D, max_norm=args.max_norm)
        self.embed = nn.Embedding(V, D)
        # word embedding
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # gru
        self.bigru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
        # hidden
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)
        # dropout
        self.dropout = nn.Dropout(args.dropout)

    def init_hidden(self, num_layers, batch_size):
        return Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim))

    def forward(self, input, hidden):
        embed = self.embed(input)
        input = embed.view(len(input), embed.size(1), -1)
        # gru
        gru_out, hidden = self.bigru(input, hidden)
        gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = F.tanh(gru_out)
        # linear
        y = self.hidden2label(gru_out)
        logit = y
        return logit, hidden