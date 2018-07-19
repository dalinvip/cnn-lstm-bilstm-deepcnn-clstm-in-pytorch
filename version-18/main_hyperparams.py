#! /usr/bin/env python
import os
import argparse
import datetime
import Config.config as configurable
import torch
import torchtext.data as data
from models import model_CNN
from models import model_HighWay_CNN
from models import model_DeepCNN
from models import model_LSTM
from models import model_BiLSTM
from models import model_CNN_LSTM
from models import model_CLSTM
from models import model_GRU
from models import model_CBiLSTM
from models import model_CGRU
from models import model_CNN_BiLSTM
from models import model_BiGRU
from models import model_CNN_BiGRU
from models import model_CNN_MUI
from models import model_DeepCNN_MUI
from models import model_BiLSTM_1
from models import model_HighWay_BiLSTM_1
import train_ALL_CNN
import train_ALL_LSTM
from DataLoader import mydatasets_self_five
from DataLoader import mydatasets_self_two
from DataLoader.load_external_word_embedding import Word_Embedding
import multiprocessing as mu
import shutil
import numpy as np
import random
# import hyperparams

# solve encoding
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)


# random seed
from DataUtils.Common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.cuda.manual_seed(seed_num)

"""
if can not fix, try this, but the speed will de slowed
#   torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
"""


# load two-classification data
def mrs_two(path, train_name, dev_name, test_name, char_data, text_field, label_field, **kargs):
    train_data, dev_data, test_data = mydatasets_self_two.MR.splits(path, train_name, dev_name, test_name,
                                                                    char_data, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data.text, min_freq=args.min_freq)
    label_field.build_vocab(train_data.label)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(args.batch_size, len(dev_data), len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter


def mrs_two_mui(path, train_name, dev_name, test_name, char_data, text_field, label_field, static_text_field, static_label_field, **kargs):
    train_data, dev_data, test_data = mydatasets_self_two.MR.splits(path, train_name, dev_name, test_name,
                                                                    char_data, text_field, label_field)
    static_train_data, static_dev_data, static_test_data = mydatasets_self_two.MR.splits(path, train_name, dev_name,
                                                                                         test_name,
                                                                                         char_data, static_text_field,
                                                                                         static_label_field)
    print("len(train_data) {} ".format(len(train_data)))
    print("len(train_data) {} ".format(len(static_train_data)))
    text_field.build_vocab(train_data, min_freq=args.min_freq)
    label_field.build_vocab(train_data)
    static_text_field.build_vocab(static_train_data, static_dev_data, static_test_data, min_freq=args.min_freq)
    static_label_field.build_vocab(static_train_data, static_dev_data, static_test_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(args.batch_size,
                                                     len(dev_data),
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter


# load five-classification data
def mrs_five(path, train_name, dev_name, test_name, char_data, text_field, label_field, **kargs):
    train_data, dev_data, test_data = mydatasets_self_five.MR.splits(path, train_name, dev_name, test_name,
                                                                     char_data, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data, min_freq=args.min_freq)
    label_field.build_vocab(train_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(args.batch_size,
                                                     len(dev_data),
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter


def mrs_five_mui(path, train_name, dev_name, test_name, char_data, text_field, label_field, static_text_field,
                 static_label_field, **kargs):
    train_data, dev_data, test_data = mydatasets_self_five.MR.splits(path, train_name, dev_name, test_name,
                                                                     char_data, text_field, label_field)
    static_train_data, static_dev_data, static_test_data = mydatasets_self_five.MR.splits(path, train_name, dev_name,
                                                                                          test_name,
                                                                                          char_data,
                                                                                         static_text_field,
                                                                                          static_label_field)
    print("len(train_data) {} ".format(len(train_data)))
    print("len(train_data) {} ".format(len(static_train_data)))
    text_field.build_vocab(train_data, min_freq=args.min_freq)
    label_field.build_vocab(train_data)
    static_text_field.build_vocab(static_train_data, static_dev_data, static_test_data, min_freq=args.min_freq)
    static_label_field.build_vocab(static_train_data, static_dev_data, static_test_data)
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
    text_field.build_vocab(train_data, dev_data, min_freq=args.min_freq)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
static_text_field = data.Field(lower=True)
static_label_field = data.Field(sequential=False)
if args.FIVE_CLASS_TASK:
    print("Executing 5 Classification Task......")
    if args.CNN_MUI is True or args.DEEP_CNN_MUI is True:
        train_iter, dev_iter, test_iter = mrs_five_mui(args.datafile_path, args.name_trainfile,
                                                       args.name_devfile, args.name_testfile, args.char_data,
                                                       text_field=text_field,label_field=label_field,
                                                       static_text_field=static_text_field,
                                                       static_label_field=static_label_field, device=-1, repeat=False,
                                                       shuffle=args.epochs_shuffle, sort=False)
    else:
        train_iter, dev_iter, test_iter = mrs_five(args.datafile_path, args.name_trainfile,
                                                   args.name_devfile, args.name_testfile, args.char_data, text_field,
                                                   label_field, device=-1, repeat=False, shuffle=args.epochs_shuffle,
                                                   sort=False)
elif args.TWO_CLASS_TASK:
    print("Executing 2 Classification Task......")
    if args.CNN_MUI is True or args.DEEP_CNN_MUI is True:
        train_iter, dev_iter, test_iter = mrs_two_mui(args.datafile_path, args.name_trainfile,
                                                      args.name_devfile, args.name_testfile, args.char_data, text_field=text_field,
                                                      label_field=label_field, static_text_field=static_text_field,
                                                      static_label_field=static_label_field, device=-1, repeat=False,
                                                      shuffle=args.epochs_shuffle, sort=False)
    else:
        train_iter, dev_iter, test_iter = mrs_two(args.datafile_path, args.name_trainfile,
                                                  args.name_devfile, args.name_testfile, args.char_data, text_field,
                                                  label_field, device=-1, repeat=False, shuffle=args.epochs_shuffle,
                                                  sort=False)

# load word2vec
if args.word_Embedding:
    word_embedding = Word_Embedding()
    if args.embed_dim is not None:
        print("word_Embedding_Path {} ".format(args.word_Embedding_Path))
        path = args.word_Embedding_Path
    print("loading word2vec vectors...")
    if args.freq_1_unk is True:
        word_vecs = word_embedding.load_my_vecs_freq1(path, text_field.vocab.itos, text_field.vocab.freqs, pro=0.5)   # has some error in this function
    else:
        word_vecs = word_embedding.load_my_vecs(path, text_field.vocab.itos, text_field.vocab.freqs, k=args.embed_dim)
        if args.CNN_MUI is True or args.DEEP_CNN_MUI is True:
            static_word_vecs = word_embedding.load_my_vecs(path, static_text_field.vocab.itos, text_field.vocab.freqs, k=args.embed_dim)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(word_vecs)))
    print("loading unknown word2vec and convert to list...")
    if args.char_data:
        print("loading unknown word by rand......")
        word_vecs = word_embedding.add_unknown_words_by_uniform(word_vecs, text_field.vocab.itos, k=args.embed_dim)
        if args.CNN_MUI is True or args.DEEP_CNN_MUI is True:
            static_word_vecs = word_embedding.add_unknown_words_by_uniform(static_word_vecs, static_text_field.vocab.itos, k=args.embed_dim)
    else:
        print("loading unknown word by avg......")
        word_vecs = word_embedding.add_unknown_words_by_avg(word_vecs, text_field.vocab.itos, k=args.embed_dim)
        if args.CNN_MUI is True or args.DEEP_CNN_MUI is True:
            static_word_vecs = word_embedding.add_unknown_words_by_avg(static_word_vecs, static_text_field.vocab.itos, k=args.embed_dim)
        print("len(word_vecs) {} ".format(len(word_vecs)))
    print("unknown word2vec loaded ! and converted to list...")


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
if args.CNN_MUI is True or args.DEEP_CNN_MUI is True:
    args.embed_num_mui = len(static_text_field.vocab)
args.cuda = (args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# save file
mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.mulu = mulu
args.save_dir = os.path.join(args.save_dir, mulu)
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

# load word2vec
if args.word_Embedding:
    args.pretrained_weight = word_vecs
    if args.CNN_MUI is True or args.DEEP_CNN_MUI is True:
        args.pretrained_weight_static = static_word_vecs

print("\nParameters:")
if os.path.exists("./Parameters.txt"):
    os.remove("./Parameters.txt")
file = open("Parameters.txt", "a")
for attr, value in sorted(args.__dict__.items()):
    if attr.upper() != "PRETRAINED_WEIGHT" and attr.upper() != "pretrained_weight_static".upper():
        print("\t{}={}".format(attr.upper(), value))
    file.write("\t{}={}\n".format(attr.upper(), value))
file.close()
shutil.copy("./Parameters.txt", "./snapshot/" + mulu + "/Parameters.txt")
shutil.copy("./hyperparams.py", "./snapshot/" + mulu)

# if args.cuda is True:
    # torch.cuda.seed()
    # torch.cuda.manual_seed(hyperparams.seed_num)
#
# model
if args.snapshot is None:
    if args.CNN:
        print("loading CNN model.....")
        model = model_CNN.CNN_Text(args)
        # save model in this time
        shutil.copy("./models/model_CNN.py", "./snapshot/" + mulu)
    elif args.DEEP_CNN:
        print("loading DEEP_CNN model......")
        model = model_DeepCNN.DEEP_CNN(args)
        shutil.copy("./models/model_DeepCNN.py", "./snapshot/" + mulu)
    elif args.DEEP_CNN_MUI:
        print("loading DEEP_CNN_MUI model......")
        model = model_DeepCNN_MUI.DEEP_CNN_MUI(args)
        shutil.copy("./models/model_DeepCNN_MUI.py", "./snapshot/" + mulu)
    elif args.LSTM:
        print("loading LSTM model......")
        model = model_LSTM.LSTM(args)
        shutil.copy("./models/model_LSTM.py", "./snapshot/" + mulu)
    elif args.GRU:
        print("loading GRU model......")
        model = model_GRU.GRU(args)
        shutil.copy("./models/model_GRU.py", "./snapshot/" + mulu)
    elif args.BiLSTM:
        print("loading BiLSTM model......")
        model = model_BiLSTM.BiLSTM(args)
        shutil.copy("./models/model_BiLSTM.py", "./snapshot/" + mulu)
    elif args.BiLSTM_1:
        print("loading BiLSTM_1 model......")
        # model = model_BiLSTM_lexicon.BiLSTM_1(args)
        model = model_BiLSTM_1.BiLSTM_1(args)
        shutil.copy("./models/model_BiLSTM_1.py", "./snapshot/" + mulu)
    elif args.CNN_LSTM:
        print("loading CNN_LSTM model......")
        model = model_CNN_LSTM.CNN_LSTM(args)
        shutil.copy("./models/model_CNN_LSTM.py", "./snapshot/" + mulu)
    elif args.CLSTM:
        print("loading CLSTM model......")
        model = model_CLSTM.CLSTM(args)
        shutil.copy("./models/model_CLSTM.py", "./snapshot/" + mulu)
    elif args.CBiLSTM:
        print("loading CBiLSTM model......")
        model = model_CBiLSTM.CBiLSTM(args)
        shutil.copy("./models/model_CBiLSTM.py", "./snapshot/" + mulu)
    elif args.CGRU:
        print("loading CGRU model......")
        model = model_CGRU.CGRU(args)
        shutil.copy("./models/model_CGRU.py", "./snapshot/" + mulu)
    elif args.CNN_BiLSTM:
        print("loading CNN_BiLSTM model......")
        model = model_CNN_BiLSTM.CNN_BiLSTM(args)
        shutil.copy("./models/model_CNN_BiLSTM.py", "./snapshot/" + mulu)
    elif args.BiGRU:
        print("loading BiGRU model......")
        model = model_BiGRU.BiGRU(args)
        shutil.copy("./models/model_BiGRU.py", "./snapshot/" + mulu)
    elif args.CNN_BiGRU:
        print("loading CNN_BiGRU model......")
        model = model_CNN_BiGRU.CNN_BiGRU(args)
        shutil.copy("./models/model_CNN_BiGRU.py", "./snapshot/" + mulu)
    elif args.CNN_MUI:
        print("loading CNN_MUI model......")
        model = model_CNN_MUI.CNN_MUI(args)
        shutil.copy("./models/model_CNN_MUI.py", "./snapshot/" + mulu)
    elif args.HighWay_CNN is True:
        print("loading HighWay_CNN model......")
        model = model_HighWay_CNN.HighWay_CNN(args)
        shutil.copy("./models/model_HighWay_CNN.py", "./snapshot/" + mulu)
    elif args.HighWay_BiLSTM_1 is True:
        print("loading HighWay_BiLSTM_1 model......")
        model = model_HighWay_BiLSTM_1.HighWay_BiLSTM_1(args)
        shutil.copy("./models/model_HighWay_BiLSTM_1.py", "./snapshot/" + mulu)
    print(model)
else:
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        model = torch.load(args.snapshot)
    except:
        print("Sorry, This snapshot doesn't exist.")
        exit()


if args.cuda is True:
    model = model.cuda()
# train or predict
if args.predict is not None:
    label = train_ALL_CNN.predict(args.predict, model, text_field, label_field)
    print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        print(test_iter)
        train_ALL_CNN.test_eval(test_iter, model, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print("\n cpu_count \n", mu.cpu_count())
    torch.set_num_threads(args.num_threads)
    if os.path.exists("./Test_Result.txt"):
        os.remove("./Test_Result.txt")
    if args.CNN:
        print("CNN training start......")
        model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, args)
    elif args.DEEP_CNN:
        print("DEEP_CNN training start......")
        model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, args)
    elif args.LSTM:
        print("LSTM training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.GRU:
        print("GRU training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.BiLSTM:
        print("BiLSTM training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.BiLSTM_1:
        print("BiLSTM_1 training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CNN_LSTM:
        print("CNN_LSTM training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CLSTM:
        print("CLSTM training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CBiLSTM:
        print("CBiLSTM training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CGRU:
        print("CGRU training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CNN_BiLSTM:
        print("CNN_BiLSTM training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.BiGRU:
        print("BiGRU training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CNN_BiGRU:
        print("CNN_BiGRU training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.CNN_MUI:
        print("CNN_MUI training start......")
        model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, args)
    elif args.DEEP_CNN_MUI:
        print("DEEP_CNN_MUI training start......")
        model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, args)
    elif args.HighWay_CNN is True:
        print("HighWay_CNN training start......")
        model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, args)
    elif args.HighWay_BiLSTM_1 is True:
        print("HighWay_BiLSTM_1 training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
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


if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Chinese NER & POS")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    args = parser.parse_args()

    config = configurable.Configurable(config_file=args.config_file)
    if config.use_cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
print("torch.cuda.initial_seed", torch.cuda.initial_seed())