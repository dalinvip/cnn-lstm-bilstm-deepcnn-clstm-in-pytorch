import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
torch.manual_seed(1233)

class  CNN_BiLSTM(nn.Module):
    
    def __init__(self, args):
        super(CNN_BiLSTM,self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.C = C
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D)
        # self.embed = nn.Embedding(V, D, max_norm=args.max_norm)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

        # CNN
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D), padding=(K//2, 0), stride=1) for K in Ks]
        print(self.convs1)

        # BiLSTM
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True, bias=True)
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)

        # linear
        L = len(Ks) * Co + self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

        # dropout
        self.dropout = nn.Dropout(args.dropout)


    def init_hidden(self,num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)),
                 Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)))


    def forward(self, x):
        embed = self.embed(x)

        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        # cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        # cnn_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # BiLSTM
        bilstm_x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, self.hidden = self.bilstm(bilstm_x, self.hidden)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        # bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)

        # CNN and BiLSTM CAT
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0)
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)

        # linear
        cnn_bilstm_out = self.hidden2label1(F.tanh(cnn_bilstm_out))
        # cnn_bilstm_out = F.tanh(self.hidden2label1(cnn_bilstm_out))
        cnn_bilstm_out = self.hidden2label2(F.tanh(cnn_bilstm_out))
        # cnn_bilstm_out = self.hidden2label2(cnn_bilstm_out)

        # output
        logit = cnn_bilstm_out
        return logit