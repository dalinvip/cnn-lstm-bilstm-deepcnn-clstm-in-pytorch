# from _sitebuiltins import _Printer
#
# import torch
# import numpy as np
#
# a = torch.ones(5)
# b = a.numpy()
# print(a)
# print(b)
# a.add_(2)
# print(a)
# print(b)
#
#
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(a)
# print(b)
# np.add(a,1,out=a)
# print(a)
# print(b)
#
# x = torch.Tensor([1,2])
# y = torch.Tensor([2,3])
# if torch.cuda.is_available():
#     x = x.cuda()
#     y = y.cuda()
#     print( x + y )
#
# from torch.autograd import Variable
# x = Variable(torch.ones(2,2),requires_grad = True)
# print("x", x)
# y = x + 2
# print("y", y)
# #
# z = y *  y * 3
# out = z.mean()
# out.backward()
# x.grad
#
# import torch
# x = torch.Tensor([1.0])
# xx = x.cuda()
# print(xx)
#
# # CUDNN TEST
# from torch.backends import cudnn
# print(cudnn.is_acceptable(xx))
#
# import torch
#
# def test1():
#     x = torch.Tensor(10, 10)
#     x_gpus = torch.cuda.comm.broadcast(x, [2, 3])
#     print([t.get_device() for t in x_gpus])
#
# def test2():
#     x = torch.Tensor(10, 10).cuda()
#     x_gpus = torch.cuda.comm.broadcast(x, [2, 3])
#     print([t.get_device() for t in x_gpus])
#
# if __name__ == '__main__':
#     test1() # -> [2, 3]
#     # test2() # -> [0, 3]
#
#
# from theano import function, config, shared, sandbox
# import theano.tensor as T
# import numpy
# import time
#
# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 1000
#
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], T.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in xrange(iters):
#     r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')

# # 给定一个字符串，判断该字符串是否满足：{对于字符串中任意两个字符c1, c2， count(c1) == count(c2)}
# str2 = "I am chinese, I love you!"
# # str = "uubbnn"
# str = "a is a"
# flag = True
# for k in str:
#     for m in str:
#         if str.count(k) == str.count(m):
#             continue
#         else:
#             print("Not satisfied")
#             flag = False
#             break
#     break
# if flag is True:
#     print("Satisfied")


# # 矩阵乘法运算，给定矩阵A，计算A的3次方，A的逆矩阵，A的行列式值，A的特征值；（可以使用相关库）
# # 参考 http://www.numpy.org/
# import  numpy as  np
# from numpy import linalg
# # 矩阵A
# A = np.array([[-1,4],[2,3]])
# print("矩阵")
# print(A)
# print(A.__class__)
#
# # A的逆矩阵
# B = linalg.inv(A)
# print("逆矩阵")
# print(B)
#
# # A转置矩阵
# C = A.T
# print("转置矩阵")
# print(C)
#
# # A三次方
# D = np.dot(A, A)
# D = np.dot(D, A)
# print("举证三次方")
# print(D)
#
# # A行列式值
# N = linalg.det(A)
# print("行列式值")
# print(N)



# 输入A矩阵，得出A矩阵的三次方
# import numpy as np
# import torch
# import scipy as sp
# from numpy import linalg
# A=[]
# print('输入矩阵维度')
# x1=input()
# x=int(x1)
# y1=input()
# y=int(y1)
# print('输入矩阵元素')
# for i in range(0,x):
#     temp=[]
#     for j in range(0,y):
#         j=input('')
#         temp.append(int(j))
#     A.append(temp)
# m = np.array(A)
# print(m)
# print(m.__class__)
# c=np.dot(m,m)
# f=np.dot(c,m)
# print('矩阵的三次方为：')
# print(f)
# A的逆矩阵
# B = linalg.inv(A)
# print("逆矩阵：")
# print(B)

# import torch
# import numpy as np

#
# import torch
# from torch.autograd import Variable
# x = torch.FloatTensor([1,2,3,4])
# print(x)
# x = Variable(x,requires_grad = True)
# print(x)
# print(x.grad)
#


# str = [1,2,3,4,5,6,7,8,9,10]
# train = str[:-4]
# print(train)
# dev = str[-4:-2]
# print(dev)
# test = str[-2:]
# print(test)
#
# for i in  range(100000):
#     x = float(input("input x"))
#     if x <= 1 and x >= -1:
#         print(x)
#     else:
#         print("error")
#         break




# import torch
# x = torch.FloatTensor(5,3,4)
# print(x)
# print(x.__class__)
# print(x.size())

# import numpy as np
# word_vecs = []
# word = "aa"
# word_vecs[word] = np.random.uniform(-0.25, 0.25, 100)
# print(word_vecs)


# for x, y in range(2), range(4):
#     print(x)

# str = "abcde"
# for i in str[1:3]:
#     print(i)

# for i,j in [range(3), range(5)]:
#     print(i)
#     print(j)
#


# dict = {0: 4, 1: 6, 2: 3, 3: 2, 4: 8, 5: 8, 6: 6}
# print(dict)

# print(dict[1])
# tuple_r_dict = lambda _dict: dict(val[::-1] for val in _dict.items())
# tuple_r_dict(tuple_r_dict(dict))
# print(dict)

# dict to list
# dict = {0: 4, 1: 6, 2: 3, 3: 2, 4: 8, 5: 8, 6: 6}
# list = []
# for index, value in dict.items():
#     list.append(value)
# print(list)

# list = [(2, "a"), (3, "b"), (4, "c"), (6, "d")]
# print(list[1])
# for i in range(len(list)):
#     print(i)
#     if 300 in list[i] :
#         print("a")
# from anaconda_navigator.utils.py3compat import cmp

# tu = [(3,"a"), (3, "a")]
# print(tu[0])
#
# if tu[0] >= tu[1]:
#     print("aa")
# # if cmp(tu[0], tu[1]):
# #     print("bb")

#
# import torch
# a = torch.FloatTensor([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
# print(a)
# b = torch.FloatTensor([[1,2,3,4,5,6,7,8],[1,2,3,4,1,2,3,4]])
# print(b)
# c = torch.mul(a, b)
# print(c)

# str = "如何 装逼 装 得 别人 一 愣 一 愣 的 ？"
# print(len(str))



# import os
# fname = './a\\w2v103100-en.100d.txt'
# print(fname + '.txt')
# if os.path.isfile(fname + ".txt"):
#     print("True")
# else:
#     print("False")

# list = []
# a = [1]
# b = [1,3]
# list += a
# print(list)
# list += b
# print(list)
# # list.append(a)
# # print(list)
# # list.append(b)
# # print(list)
# # list.extend(a)
# # print(list)
# # list.extend(b)
# # print(list)

# pos_u = [([0, 1, 3, 3], 2), ([1, 2, 3, 4], 3), ([2, 3, 4, 5], 3)]
# a = []
#
# print(len(pos_u))
# for i in range(len(pos_u)):
#     print((pos_u[i])[0])
#     sum = 0
#     for j in range(len(pos_u[i][0])):
#         sum += pos_u[i][0][j]
#     a.append((sum,pos_u[i][1]))
# print(a)

# for i in range(2,1):
#     print("s")
# for j in range(2,2):
#     print(j)

# for i in range(4, 0, -1):
##      print(i)

# import pickle

# if " " != " ":
#     print("aa")

# import numpy as np
# # a = 1.12509999999999999
# b = []
# a = -0.947567
# print(round(a,6))
# print(np.round(a,6))
# c = np.round(a,6)
# b.append(c)
# print(c)


# # 不一致无法转换
# # 不一致无法转换
# list = [[0.111,0.222,44.333,15.11],[0.111,0.222542],[0.1178741,0.222,0.33304540,0.1]]
# a = np.array(list)
# b = torch.from_numpy(a)
# print(b)

# f =open("./word2vec/glove.6B.100d.txt",encoding="utf-8")
# lines = f.readlines()[1:]
# count = 0
# for line in lines:
#     values = line.split(" ")
#     s = 0
#     for count, val in enumerate(values):
#         if count == 0:
#             continue
#         s += 1
#     if s != 100:
#         print("aaaa")

# import torch
# from torch.autograd import Variable
# a = torch.FloatTensor(10,10,10)
# a = Variable(a)
# print(a.__class__)
# print(a[-1].__class__)

# import torch
# from torch.autograd import  Variable
# from torch.utils.data import TensorDataset, DataLoader
# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
# # print(x)
# # print(y)
#
# loader = DataLoader(TensorDataset(x, y), batch_size=8)
#
# print(loader)
#
# for epoch in range(1):
#     for x_batch, y_batch in loader:
#         print(x_batch)
#         print(y_batch)
#         break
#         x_var, y_var = Variable(x), Variable(y)
#




# import torch.nn as nn
# from torch.autograd import Variable
# emd = nn.Embedding(10, 3)
# input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
# print(input)
# emd = emd(input)
# print(emd)
#
# embedding = nn.Embedding(10, 3, padding_idx=1)
# input = Variable(torch.LongTensor([[1,2,1,5],[0,2,0,5]]))
# print(input)
# a = embedding(input)
# print(a)
# # padding_idx=1 遇到1输出的时候都用0填充
# embedding = nn.Embedding(10, 3, padding_idx=0)
# input = Variable(torch.LongTensor([[1,2,1,5],[0,2,0,5]]))
# print(input)
# a = embedding(input)
# print(a)

# import torch
# x = torch.randn(2, 2, 3)
# print(x)
# y = torch.transpose(x, 0, 1)
# print(y)
#
#

# import torch
# x = torch.Tensor([1, 2, 3, 4])
# print(x)
# a = x.unsqueeze(0)
# print(a)
# b = x.unsqueeze(1)
# print(b)

# import torch
# x = torch.Tensor([1, 2, 3, 4])
# print(x.__call__)
#


# list = [1,2,3,5,6,4]
# for i in range(10):
#     # print(list[:i])
#     print(min(list[:,1]))
#



# import numpy as np
# x = np.random.rand(3,5)
# print(x)
# print("***********************************")
# print(x[:2])
# print(x[1:2])
# print("//////////////////////")
# print(x[:,2])
# print(x[1:,2])




# import numpy as np
# k=2
# n=3
# a = np.zeros((k,n))
# print(a)
# print(a.__class__)
# b = np.mat(a)
# print(b)
# print(b.__class__)
# c = np.zeros(k)
# print(c)


# import torch
# a = torch.randn(64, 100, 150)
# print(a.size())
# b = torch.cat(a, 0)
# print(b.size())


# import torch
# t= torch.Tensor([[1,2],[3,4]])
# print(t)
# a = torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
# print(a)
# t= torch.Tensor([[1,2],[3,4]])
# b = torch.gather(t, 0, torch.LongTensor([[0,1],[1,0]]))
# print(b)
# t= torch.Tensor([[1,2],[3,4]])
# c = torch.gather(t, 1, torch.LongTensor([[0,0],[0,0]]))
# print(c)
# t= torch.Tensor([[1,2],[3,4]])
# d = torch.gather(t, 0, torch.LongTensor([[0,0],[0,0]]))
# print(d)


# import torch
# import torch.nn as nn
# import torch.autograd as autograd
# m = nn.BatchNorm1d(100)
# # Without Learnable Parameters
# m = nn.BatchNorm1d(100, affine=False)
# input = autograd.Variable(torch.randn(20, 100))
# print(input)
# output = m(input)
# print(output)
#


# flaf =True
# if flaf:
#     print()

# ex = ["john", "love", "marry"]
# a = []
# print(ex[1][1])
# print(len(ex))
# print(len(ex[1]))
# for i in range(len(ex)):
#     for j in range(len(ex[i])):
#         a += ex[i][j]
# print(a)
#



