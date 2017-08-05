#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
from Cython.Shadow import profile
from sklearn.utils import shuffle

import sstdatasets as sstdatasets
import model_CNN
import model_DeepCNN
import model_LSTM
import model_BiLSTM
import model_BiLSTM_1
import model_CNN_LSTM
import model_CLSTM
import model_GRU
import model_CBiLSTM
import model_CGRU
import model_CNN_BiLSTM
import model_BiGRU
import train
import train_CNN
import train_DeepCNN
import train_LSTM
import train_BiLSTM
import train_BiLSTM_1
import train_CNN_LSTM
import train_CLSTM
import train_GRU
import train_CGRU
import train_CNN_BiLSTM
import train_CBiLSTM
import train_BiGRU
import mydatasets
import mydatasets_self_five
import mydatasets_self_two
import multiprocessing as mu
import shutil
import numpy as np
import hyperparams
# solve encoding
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(121)

parser = argparse.ArgumentParser(description="text classification")
# learning
parser.add_argument('-lr', type=float, default=hyperparams.learning_rate, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=hyperparams.epochs, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=hyperparams.batch_size, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=hyperparams.log_interval,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=hyperparams.test_interval, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=hyperparams.save_interval, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default=hyperparams.save_dir, help='where to save the snapshot')
# data
parser.add_argument('-datafile_path', type=str, default=hyperparams.datafile_path, help='datafile path')
parser.add_argument('-name_trainfile', type=str, default=hyperparams.name_trainfile, help='train file name')
parser.add_argument('-name_devfile', type=str, default=hyperparams.name_devfile, help='dev file name')
parser.add_argument('-name_testfile', type=str, default=hyperparams.name_testfile, help='test file name')
parser.add_argument('-word_data', action='store_true', default=hyperparams.word_data, help='whether to use CNN model')
parser.add_argument('-char_data', action='store_true', default=hyperparams.char_data, help='whether to use CNN model')
parser.add_argument('-shuffle', action='store_true', default=hyperparams.shuffle, help='shuffle the data every epoch' )
# task select
parser.add_argument('-FIVE_CLASS_TASK', action='store_true', default=hyperparams.FIVE_CLASS_TASK, help='whether to execute five-classification-task')
parser.add_argument('-TWO_CLASS_TASK', action='store_true', default=hyperparams.TWO_CLASS_TASK, help='whether to execute two-classification-task')
# model
parser.add_argument('-dropout', type=float, default=hyperparams.dropout, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=hyperparams.max_norm, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=hyperparams.embed_dim, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=hyperparams.kernel_num, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default=hyperparams.kernel_sizes, help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=hyperparams.static, help='fix the embedding')
parser.add_argument('-CNN', action='store_true', default=hyperparams.CNN, help='whether to use CNN model')
parser.add_argument('-DEEP_CNN', action='store_true', default=hyperparams.DEEP_CNN, help='whether to use Depp CNN model')
parser.add_argument('-LSTM', action='store_true', default=hyperparams.LSTM, help='whether to use LSTM model')
parser.add_argument('-GRU', action='store_true', default=hyperparams.GRU, help='whether to use GRU model')
parser.add_argument('-BiLSTM', action='store_true', default=hyperparams.BiLSTM, help='whether to use Bi-LSTM model')
parser.add_argument('-BiLSTM_1', action='store_true', default=hyperparams.BiLSTM_1, help='whether to use Bi-LSTM_1 model')
parser.add_argument('-CNN_LSTM', action='store_true', default=hyperparams.CNN_LSTM, help='whether to use CNN_LSTM model')
parser.add_argument('-CNN_BiLSTM', action='store_true', default=hyperparams.CNN_BiLSTM, help='whether to use CNN_BiLSTM model')
parser.add_argument('-CLSTM', action='store_true', default=hyperparams.CLSTM, help='whether to use CLSTM model')
parser.add_argument('-CBiLSTM', action='store_true', default=hyperparams.CBiLSTM, help='whether to use CBiLSTM model')
parser.add_argument('-CGRU', action='store_true', default=hyperparams.CGRU, help='whether to use CGRU model')
parser.add_argument('-BiGRU', action='store_true', default=hyperparams.BiGRU, help='whether to use BiGRU model')
parser.add_argument('-word_Embedding', action='store_true', default=hyperparams.word_Embedding, help='whether to load word embedding')
parser.add_argument('-lstm-hidden-dim', type=int, default=hyperparams.lstm_hidden_dim, help='the number of embedding dimension in LSTM hidden layer')
parser.add_argument('-lstm-num-layers', type=int, default=hyperparams.lstm_num_layers, help='the number of embedding dimension in LSTM hidden layer')
# device
parser.add_argument('-device', type=int, default=hyperparams.device, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no_cuda', action='store_true', default=hyperparams.no_cuda, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=hyperparams.snapshot, help='filename of model snapshot [default: None]')
# parser.add_argument('-snapshot', type=str, default="./snapshot/2017-07-13_07-26-41/snapshot_steps155000.pt", help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=hyperparams.predict, help='predict the sentence given')
# parser.add_argument('-predict', type=str, default="I love you so much， and I love you forever ", help='predict the sentence given')
# parser.add_argument('-predict', type=str, default="I hate you  and I hate you so sad, I am crying ", help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=hyperparams.test, help='train or test')
args = parser.parse_args()


# load SST dataset
def sst(text_field, label_field,  **kargs):
    print("SST")
    train_data, dev_data, test_data = sstdatasets.SST.splits(text_field, label_field, fine_grained=True)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter 



# load two-classification data
def mrs_two(path, train_name, dev_name, test_name, char_data, text_field, label_field, **kargs):
    train_data, dev_data, test_data = mydatasets_self_two.MR.splits(path, train_name, dev_name, test_name, char_data, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data)
    label_field.build_vocab(train_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(args.batch_size,
                                                     len(dev_data),
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter

# load five-classification data
def mrs_five(path, train_name, dev_name, test_name, char_data, text_field, label_field, **kargs):
    train_data, dev_data, test_data = mydatasets_self_five.MR.splits(path, train_name, dev_name, test_name, char_data, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data)
    label_field.build_vocab(train_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(args.batch_size,
                                                     len(dev_data),
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter


# load word embedding
def load_my_vecs(path, vocab):
    word_vecs = {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            values = line.split(" ")
            word = values[0]
            if word in vocab:  #whehter to judge if in vocab
                vector = []
                for count, val in enumerate(values):
                    if count == 0:
                        continue
                    vector.append(float(val))
                word_vecs[word] = vector
    return word_vecs

# solve unknown by avg word embedding
def add_unknown_words_by_avg(word_vecs, vocab, k=100):
    # solve unknown words inplaced by zero list
    word_vecs_numpy = []
    for word in vocab:
        if word in word_vecs:
            word_vecs_numpy.append(word_vecs[word])
    print(len(word_vecs_numpy))
    col = []
    for i in range(k):
        sum = 0.0
        for j in range(int(len(word_vecs_numpy) / 4)):
            sum += word_vecs_numpy[j][i]
            sum = round(sum, 6)
        col.append(sum)
    zero = []
    for m in range(k):
        avg = col[m] / (len(col) * 3)
        avg = round(avg, 6)
        zero.append(float(avg))

    list_word2vec = []
    oov = 0
    iov = 0
    for word in vocab:
        if word not in word_vecs:
            # word_vecs[word] = np.random.uniform(-0.25, 0.25, k).tolist()
            # word_vecs[word] = [0.0] * k
            oov += 1
            word_vecs[word] = zero
            list_word2vec.append(word_vecs[word])
        else:
            iov += 1
            list_word2vec.append(word_vecs[word])
    print("oov count", oov)
    print("iov count", iov)
    return list_word2vec

# solve unknown word by uniform(-0.25,0.25)
def add_unknown_words_by_uniform(word_vecs, vocab, k=100):
    list_word2vec = []
    oov = 0
    iov = 0
    # uniform = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
    for word in vocab:
        if word not in word_vecs:
            oov += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
            # word_vecs[word] = uniform
            list_word2vec.append(word_vecs[word])
        else:
            iov += 1
            list_word2vec.append(word_vecs[word])
    print("oov count", oov)
    print("iov count", iov)
    return list_word2vec

# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
if args.FIVE_CLASS_TASK:
    print("Executing 5 Classification Task......")
    train_iter, dev_iter, test_iter = mrs_five(args.datafile_path, args.name_trainfile,
                                               args.name_devfile, args.name_testfile, args.char_data, text_field, label_field, device=-1, repeat=False)
elif args.TWO_CLASS_TASK:
    print("Executing 2 Classification Task......")
    train_iter, dev_iter, test_iter = mrs_two(args.datafile_path, args.name_trainfile,
                                              args.name_devfile, args.name_testfile, args.char_data, text_field, label_field, device=-1, repeat=False)



# load word2vec
if args.word_Embedding:
    if args.embed_dim == 100:
        path = "./word2vec/glove.6B.100d.txt"
    elif args.embed_dim == 200:
        path = "./word2vec/glove.6B.200d.txt"
    elif args.embed_dim == 300:
        path = "./word2vec/glove.6B.300d.txt"
    print("loading word2vec vectors...")
    print("len(text_field.vocab.itos)", len(text_field.vocab.itos))
    word_vecs = load_my_vecs(path, text_field.vocab.itos)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(word_vecs)))
    print("loading unknown word2vec and convert to list...")
    # word_vecs = add_unknown_words_by_avg(word_vecs, text_field.vocab.itos, k=args.embed_dim)
    word_vecs = add_unknown_words_by_uniform(word_vecs, text_field.vocab.itos, k=args.embed_dim)
    print("unknown word2vec loaded ! and converted to list...")


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# save file
mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.save_dir = os.path.join(args.save_dir, mulu)

# load word2vec
if args.word_Embedding:
    args.pretrained_weight = word_vecs

print("\nParameters:")
if os.path.exists("./Parameters.txt"):
    os.remove("./Parameters.txt")
file = open("Parameters.txt", "a")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
    file.write("\t{}={}\n".format(attr.upper(), value))
file.close()


# model
if args.snapshot is None:
    if args.CNN:
        print("loading CNN model.....")
        model = model_CNN.CNN_Text(args)
    elif args.DEEP_CNN:
        print("loading DEEP_CNN model......")
        model = model_DeepCNN.DEEP_CNN(args)
    elif args.LSTM:
        print("loading LSTM model......")
        model = model_LSTM.LSTM(args)
    elif args.GRU:
        print("loading GRU model......")
        model = model_GRU.GRU(args)
    elif args.BiLSTM:
        print("loading BiLSTM model......")
        model = model_BiLSTM.BiLSTM(args)
    elif args.BiLSTM_1:
        print("loading BiLSTM_1 model......")
        model = model_BiLSTM_1.BiLSTM_1(args)
    elif args.CNN_LSTM:
        print("loading CNN_LSTM model......")
        model = model_CNN_LSTM.CNN_LSTM(args)
    elif args.CLSTM:
        print("loading CLSTM model......")
        model = model_CLSTM.CLSTM(args)
    elif args.CBiLSTM:
        print("loading CBiLSTM model......")
        model = model_CBiLSTM.CBiLSTM(args)
    elif args.CGRU:
        print("loading CGRU model......")
        model = model_CGRU.CGRU(args)
    elif args.CNN_BiLSTM:
        print("loading CNN_BiLSTM model......")
        model = model_CNN_BiLSTM.CNN_BiLSTM(args)
    elif args.BiGRU:
        print("loading BiGRU model......")
        model = model_BiGRU.BiGRU(args)
    print(model)
else:
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        model = torch.load(args.snapshot)
    except:
        print("Sorry, This snapshot doesn't exist.")
        exit()
        

# train or predict
if args.predict is not None:
    label = train_CNN.predict(args.predict, model, text_field, label_field)
    print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        print(test_iter)
        train.test_eval(test_iter, model, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print("\n cpu_count \n", mu.cpu_count())
    torch.set_num_threads(4)
    if os.path.exists("./Test_Result.txt"):
        os.remove("./Test_Result.txt")
    if args.CNN:
        print("CNN training start......")
        model_count = train_CNN.train(train_iter, dev_iter, test_iter, model, args)
    elif args.DEEP_CNN:
        print("DEEP_CNN training start......")
        model_count = train_DeepCNN.train(train_iter, dev_iter, test_iter, model, args)
    elif args.LSTM:
        print("LSTM training start......")
        model_count = train_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.GRU:
        print("GRU training start......")
        model_count = train_GRU.train(train_iter, dev_iter, test_iter, model, args)
    elif args.BiLSTM:
        print("BiLSTM training start......")
        model_count = train_BiLSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.BiLSTM_1:
        print("BiLSTM_1 training start......")
        model_count = train_BiLSTM_1.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CNN_LSTM:
        print("CNN_LSTM training start......")
        model_count = train_CNN_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CLSTM:
        print("CLSTM training start......")
        model_count = train_CLSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CBiLSTM:
        print("CBiLSTM training start......")
        model_count = train_CBiLSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CGRU:
        print("CGRU training start......")
        model_count = train_CGRU.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CNN_BiLSTM:
        print("CNN_BiLSTM training start......")
        model_count = train_CNN_BiLSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.BiGRU:
        print("BiGRU training start......")
        model_count = train_BiGRU.train(train_iter, dev_iter, test_iter, model, args)
    print("Model_count", model_count)
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
        file.write("\n")
        file.close()
        shutil.copy("./Test_Result.txt", "./snapshot/" + mulu + "/Test_Result.txt")
        shutil.copy("./Parameters.txt", "./snapshot/" + mulu + "/Parameters.txt")


