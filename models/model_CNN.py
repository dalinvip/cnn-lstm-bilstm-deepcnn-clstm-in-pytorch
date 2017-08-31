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
class  CNN_Text(nn.Module):
    
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
            self.embed = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True)
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
                                     padding=(K//2, 0), dilation=1, bias=True) for K in Ks]
        else:
            print("using narrow convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
        # self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K // 2, 0)) for K in Ks]
        print(self.convs1)

        # for con in self.convs1:
            # print("PP {} ".format(con.weight))
        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                # conv.weight.data.fill_(1)
                # conv.bias.data.fill_(0)

                # init.xavier_uniform(conv.weight, gain=np.sqrt(2.0))
                # init.constant(conv.bias, 0.1)  # many prople use it why I has not good
                # init.uniform(conv.bias, 0.1, 0.1)

                # init.xavier_normal(conv.weight)
                # init.xavier_normal(conv.bias)

                # n = args.kernel_sizes[0] * args.kernel_sizes[1]
                # conv.weight.data.normal_(0, np.sqrt(2. / n))

                # print("asasas {}".format(conv.weight.data.__class__))

                init.xavier_normal(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)
                print(" in {} out {} ".format(fan_in, fan_out))
                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
                print("aaaaaaaaaaaaa {} ".format(std))
                init.uniform(conv.bias, 0, 0)


                # init.xavier_uniform(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                # fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)
                # print(" in {} out {} ".format(fan_in, fan_out))
                # std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
                # a = np.sqrt(3.0) * std
                # print("aaaaaaaaaaaaa {} ".format(a))

                # if isinstance(conv.weight.data, Variable):
                #     print("ssssssssssssssssssssssssssssssssss")
                # else:
                #     print("eeeeeeeeeeeeeeeeeeeeeeee")
                #     fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)
                #     print(" in {} out {} ".format(fan_in, fan_out))
                #     std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
                #     print("aaaaaaaaaaaaa {} ".format(std))
                # init.xavier_normal(conv.weight)
                # init.xavier_uniform(conv.weight)
                # init.uniform(conv.bias)
                # init.constant(conv.bias, val=0)
                # init.xavier_normal(conv.weight, gain=args.init_weight_value)

                # print("QQ {} ".format(conv.weight))
                # print("QQ {} ".format(conv.bias))

        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        # self.dropout = nn.Dropout2d(args.dropout)
        # self.dropout = nn.AlphaDropout(args.dropout)
        in_fea = len(Ks) * Co
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)
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
        x = self.dropout(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        if self.args.batch_normalizations is True:
            x = [self.convs1_bn(F.relu(conv(x))).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            # x = [F.tanh(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            # x = [conv(x).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N,len(Ks)*Co)
        if self.args.batch_normalizations is True:
            x = self.fc1_bn(self.fc1(x))
            logit = self.fc2_bn(self.fc2(F.relu(x)))

            # x = self.fc1_bn(self.fc1(x))
            # # x = self.fc1(x)
            # logit = self.fc2_bn(self.fc2(F.relu(x)))
            # logit = self.fc2_bn(self.fc2(F.relu(x)))

            # x = self.fc1(x)
            # logit = self.fc2(x)

            # x = F.relu(self.fc1_bn(self.fc1(x)))
            # logit = self.fc2(x)

            # x = F.relu(self.fc1(x))
            # logit = self.fc2_bn(self.fc2(x))

            # x = F.relu(self.fc1(F.relu(x)))
            # x= self.fc1(x)
            # logit = self.fc2(x)

        else:
            # logit = self.fc1(F.tanh(x)) # (N,C)

            # x = self.fc1(F.relu(x))
            # x = self.dropout(x)
            x = self.fc1(x)
            logit = self.fc2(F.relu(x))
            # logit = self.fc2(F.tanh(x))

            # x = self.fc1(x)
            # x = self.dropout(x)
            # logit = self.fc2(x)

            # x = F.relu(self.fc1(x))
            # # x = self.fc1(x)
            # logit = self.fc2(x)
        # print("self.embed.weight {} ".format(self.embed.weight))
        return logit