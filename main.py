# coding=utf-8
# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : main.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import os
import argparse
import datetime
import Config.config as configurable
import torch
import torchtext.data as data
from models.model_CNN import CNN_Text
from models.model_HighWay_CNN import HighWay_CNN
from models.model_DeepCNN import DEEP_CNN
from models.model_LSTM import LSTM
from models.model_BiLSTM import BiLSTM
from models.model_CNN_LSTM import CNN_LSTM
from models.model_CLSTM import CLSTM
from models.model_GRU import GRU
from models.model_CBiLSTM import CBiLSTM
from models.model_CGRU import CGRU
from models.model_CNN_BiLSTM import CNN_BiLSTM
from models.model_BiGRU import BiGRU
from models.model_CNN_BiGRU import CNN_BiGRU
from models.model_CNN_MUI import CNN_MUI
from models.model_DeepCNN_MUI import DEEP_CNN_MUI
from models.model_BiLSTM_1 import BiLSTM_1
from models.model_HighWay_BiLSTM_1 import HighWay_BiLSTM_1
import train_ALL_CNN
import train_ALL_LSTM
from DataLoader import mydatasets_self_five
from DataLoader import mydatasets_self_two
from DataUtils.Load_Pretrained_Embed import load_pretrained_emb_zeros, load_pretrained_emb_avg, load_pretrained_emb_Embedding, load_pretrained_emb_uniform
import multiprocessing as mu
import shutil
import numpy as np
import random

# solve encoding
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
from DataUtils.Common import seed_num, pad, unk
torch.manual_seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.cuda.manual_seed(seed_num)


def mrs_two(path, train_name, dev_name, test_name, char_data, text_field, label_field, **kargs):
    """
    :function: load two-classification data
    :param path:
    :param train_name: train path
    :param dev_name: dev path
    :param test_name: test path
    :param char_data: char data
    :param text_field: text dict for finetune
    :param label_field: label dict for finetune
    :param kargs: others arguments
    :return: batch train, batch dev, batch test
    """
    train_data, dev_data, test_data = mydatasets_self_two.MR.splits(path, train_name, dev_name, test_name, char_data, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data.text, min_freq=config.min_freq)
    # text_field.build_vocab(train_data.text, dev_data.text, test_data.text, min_freq=config.min_freq)
    label_field.build_vocab(train_data.label)
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),batch_sizes=(config.batch_size, len(dev_data), len(test_data)), **kargs)
    return train_iter, dev_iter, test_iter


def mrs_two_mui(path, train_name, dev_name, test_name, char_data, text_field, label_field, static_text_field, static_label_field, **kargs):
    """
    :function: load two-classification data
    :param path:
    :param train_name: train path
    :param dev_name: dev path
    :param test_name: test path
    :param char_data: char data
    :param text_field: text dict for finetune
    :param label_field: label dict for finetune
    :param static_text_field: text dict for static(no finetune)
    :param static_label_field: label dict for static(no finetune)
    :param kargs: others arguments
    :return: batch train, batch dev, batch test
    """
    train_data, dev_data, test_data = mydatasets_self_two.MR.splits(path, train_name, dev_name, test_name, char_data, text_field, label_field)
    static_train_data, static_dev_data, static_test_data = mydatasets_self_two.MR.splits(path, train_name, dev_name, test_name,char_data, static_text_field, static_label_field)
    print("len(train_data) {} ".format(len(train_data)))
    print("len(static_train_data) {} ".format(len(static_train_data)))
    text_field.build_vocab(train_data, min_freq=config.min_freq)
    label_field.build_vocab(train_data)
    static_text_field.build_vocab(static_train_data, static_dev_data, static_test_data, min_freq=config.min_freq)
    static_label_field.build_vocab(static_train_data, static_dev_data, static_test_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data), batch_sizes=(config.batch_size, len(dev_data), len(test_data)), **kargs)
    return train_iter, dev_iter, test_iter


# load five-classification data
def mrs_five(path, train_name, dev_name, test_name, char_data, text_field, label_field, **kargs):
    """
    :function: load five-classification data
    :param path:
    :param train_name: train path
    :param dev_name: dev path
    :param test_name: test path
    :param char_data: char data
    :param text_field: text dict for finetune
    :param label_field: label dict for finetune
    :param kargs: others arguments
    :return: batch train, batch dev, batch test
    """
    train_data, dev_data, test_data = mydatasets_self_five.MR.splits(path, train_name, dev_name, test_name, char_data, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data, min_freq=config.min_freq)
    label_field.build_vocab(train_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data), batch_sizes=(config.batch_size, len(dev_data), len(test_data)), **kargs)
    return train_iter, dev_iter, test_iter


def mrs_five_mui(path, train_name, dev_name, test_name, char_data, text_field, label_field, static_text_field, static_label_field, **kargs):
    """
    :function: load five-classification data
    :param path:
    :param train_name: train path
    :param dev_name: dev path
    :param test_name: test path
    :param char_data: char data
    :param text_field: text dict for finetune
    :param label_field: label dict for finetune
    :param static_text_field: text dict for static(no finetune)
    :param static_label_field: label dict for static(no finetune)
    :param kargs: others arguments
    :return: batch train, batch dev, batch test
    """
    train_data, dev_data, test_data = mydatasets_self_five.MR.splits(path, train_name, dev_name, test_name, char_data, text_field, label_field)
    static_train_data, static_dev_data, static_test_data = mydatasets_self_five.MR.splits(path, train_name, dev_name, test_name, char_data, static_text_field, static_label_field)
    print("len(train_data) {} ".format(len(train_data)))
    print("len(static_train_data) {} ".format(len(static_train_data)))
    text_field.build_vocab(train_data, min_freq=config.min_freq)
    label_field.build_vocab(train_data)
    static_text_field.build_vocab(static_train_data, static_dev_data, static_test_data, min_freq=config.min_freq)
    static_label_field.build_vocab(static_train_data, static_dev_data, static_test_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data), batch_sizes=(config.batch_size, len(dev_data), len(test_data)), **kargs)
    return train_iter, dev_iter, test_iter


def load_preEmbedding():
    # load word2vec
    static_pretrain_embed = None
    pretrain_embed = None
    if config.word_Embedding:
        print("word_Embedding_Path {} ".format(config.word_Embedding_Path))
        path = config.word_Embedding_Path
        print("loading pretrain embedding......")
        paddingkey = pad
        pretrain_embed = load_pretrained_emb_avg(path=path, text_field_words_dict=config.text_field.vocab.itos,
                                                       pad=paddingkey)
        if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
            static_pretrain_embed = load_pretrained_emb_avg(path=path, text_field_words_dict=config.static_text_field.vocab.itos,
                                                                  pad=paddingkey)
        config.pretrained_weight = pretrain_embed
        if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
            config.pretrained_weight_static = static_pretrain_embed

        print("pretrain embedding load finished!")


def Load_Data():
    """
    load five classification task data and two classification task data
    :return:
    """
    train_iter, dev_iter, test_iter = None, None, None
    if config.FIVE_CLASS_TASK:
        print("Executing 5 Classification Task......")
        if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
            train_iter, dev_iter, test_iter = mrs_five_mui(config.datafile_path, config.name_trainfile, config.name_devfile, config.name_testfile, config.char_data, text_field=config.text_field, label_field=config.label_field,
                                                           static_text_field=config.static_text_field, static_label_field=config.static_label_field, device=-1, repeat=False, shuffle=config.epochs_shuffle, sort=False)
        else:
            train_iter, dev_iter, test_iter = mrs_five(config.datafile_path, config.name_trainfile, config.name_devfile, config.name_testfile, config.char_data,
                                                       config.text_field, config.label_field, device=-1, repeat=False, shuffle=config.epochs_shuffle, sort=False)
    elif config.TWO_CLASS_TASK:
        print("Executing 2 Classification Task......")
        if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
            train_iter, dev_iter, test_iter = mrs_two_mui(config.datafile_path, config.name_trainfile, config.name_devfile, config.name_testfile, config.char_data, text_field=config.text_field, label_field=config.label_field,
                                                          static_text_field=config.static_text_field, static_label_field=config.static_label_field, device=-1, repeat=False, shuffle=config.epochs_shuffle, sort=False)
        else:
            train_iter, dev_iter, test_iter = mrs_two(config.datafile_path, config.name_trainfile, config.name_devfile, config.name_testfile, config.char_data, config.text_field,
                                                      config.label_field, device=-1, repeat=False, shuffle=config.epochs_shuffle, sort=False)

    return train_iter, dev_iter, test_iter


def define_dict():
    """
     use torchtext to define word and label dict
    """
    print("use torchtext to define word dict......")
    config.text_field = data.Field(lower=True)
    config.label_field = data.Field(sequential=False)
    config.static_text_field = data.Field(lower=True)
    config.static_label_field = data.Field(sequential=False)
    print("use torchtext to define word dict finished.")
    # return text_field


def save_arguments():
    shutil.copytree("./Config", "./snapshot/" + config.mulu + "/Config")


def update_arguments():
    config.lr = config.learning_rate
    config.init_weight_decay = config.weight_decay
    config.init_clip_max_norm = config.clip_max_norm
    config.embed_num = len(config.text_field.vocab)
    config.class_num = len(config.label_field.vocab) - 1
    config.paddingId = config.text_field.vocab.stoi[pad]
    config.unkId = config.text_field.vocab.stoi[unk]
    if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
        config.embed_num_mui = len(config.static_text_field.vocab)
        config.paddingId_mui = config.static_text_field.vocab.stoi[pad]
        config.unkId_mui = config.static_text_field.vocab.stoi[unk]
    # config.kernel_sizes = [int(k) for k in config.kernel_sizes.split(',')]
    print(config.kernel_sizes)
    mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.mulu = mulu
    config.save_dir = os.path.join(""+config.save_dir, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)


def load_model():
    model = None
    if config.snapshot is None:
        if config.CNN:
            print("loading CNN model.....")
            model = CNN_Text(config)
            # save model in this time
            shutil.copy("./models/model_CNN.py", "./snapshot/" + config.mulu)
        elif config.DEEP_CNN:
            print("loading DEEP_CNN model......")
            model = DEEP_CNN(config)
            shutil.copy("./models/model_DeepCNN.py", "./snapshot/" + config.mulu)
        elif config.DEEP_CNN_MUI:
            print("loading DEEP_CNN_MUI model......")
            model = DEEP_CNN_MUI(config)
            shutil.copy("./models/model_DeepCNN_MUI.py", "./snapshot/" + config.mulu)
        elif config.LSTM:
            print("loading LSTM model......")
            model = LSTM(config)
            shutil.copy("./models/model_LSTM.py", "./snapshot/" + config.mulu)
        elif config.GRU:
            print("loading GRU model......")
            model = GRU(config)
            shutil.copy("./models/model_GRU.py", "./snapshot/" + config.mulu)
        elif config.BiLSTM:
            print("loading BiLSTM model......")
            model = BiLSTM(config)
            shutil.copy("./models/model_BiLSTM.py", "./snapshot/" + config.mulu)
        elif config.BiLSTM_1:
            print("loading BiLSTM_1 model......")
            # model = model_BiLSTM_lexicon.BiLSTM_1(config)
            model = BiLSTM_1(config)
            shutil.copy("./models/model_BiLSTM_1.py", "./snapshot/" + config.mulu)
        elif config.CNN_LSTM:
            print("loading CNN_LSTM model......")
            model = CNN_LSTM(config)
            shutil.copy("./models/model_CNN_LSTM.py", "./snapshot/" + config.mulu)
        elif config.CLSTM:
            print("loading CLSTM model......")
            model = CLSTM(config)
            shutil.copy("./models/model_CLSTM.py", "./snapshot/" + config.mulu)
        elif config.CBiLSTM:
            print("loading CBiLSTM model......")
            model = CBiLSTM(config)
            shutil.copy("./models/model_CBiLSTM.py", "./snapshot/" + config.mulu)
        elif config.CGRU:
            print("loading CGRU model......")
            model = CGRU(config)
            shutil.copy("./models/model_CGRU.py", "./snapshot/" + config.mulu)
        elif config.CNN_BiLSTM:
            print("loading CNN_BiLSTM model......")
            model = CNN_BiLSTM(config)
            shutil.copy("./models/model_CNN_BiLSTM.py", "./snapshot/" + config.mulu)
        elif config.BiGRU:
            print("loading BiGRU model......")
            model = BiGRU(config)
            shutil.copy("./models/model_BiGRU.py", "./snapshot/" + config.mulu)
        elif config.CNN_BiGRU:
            print("loading CNN_BiGRU model......")
            model = CNN_BiGRU(config)
            shutil.copy("./models/model_CNN_BiGRU.py", "./snapshot/" + config.mulu)
        elif config.CNN_MUI:
            print("loading CNN_MUI model......")
            model = CNN_MUI(config)
            shutil.copy("./models/model_CNN_MUI.py", "./snapshot/" + config.mulu)
        elif config.HighWay_CNN is True:
            print("loading HighWay_CNN model......")
            model = HighWay_CNN(config)
            shutil.copy("./models/model_HighWay_CNN.py", "./snapshot/" + config.mulu)
        elif config.HighWay_BiLSTM_1 is True:
            print("loading HighWay_BiLSTM_1 model......")
            model = HighWay_BiLSTM_1(config)
            shutil.copy("./models/model_HighWay_BiLSTM_1.py", "./snapshot/" + config.mulu)
        print(model)
    else:
        print('\nLoading model from [%s]...' % config.snapshot)
        try:
            model = torch.load(config.snapshot)
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()
    if config.cuda is True:
        model = model.cuda()
    return model


def start_train(model, train_iter, dev_iter, test_iter):
    """
    :function：start train
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    if config.predict is not None:
        label = train_ALL_CNN.predict(config.predict, model, config.text_field, config.label_field)
        print('\n[Text]  {}[Label] {}\n'.format(config.predict, label))
    elif config.test:
        try:
            print(test_iter)
            train_ALL_CNN.test_eval(test_iter, model, config)
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print("\n cpu_count \n", mu.cpu_count())
        torch.set_num_threads(config.num_threads)
        if os.path.exists("./Test_Result.txt"):
            os.remove("./Test_Result.txt")
        if config.CNN:
            print("CNN training start......")
            model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, config)
        elif config.DEEP_CNN:
            print("DEEP_CNN training start......")
            model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, config)
        elif config.LSTM:
            print("LSTM training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.GRU:
            print("GRU training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.BiLSTM:
            print("BiLSTM training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.BiLSTM_1:
            print("BiLSTM_1 training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.CNN_LSTM:
            print("CNN_LSTM training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.CLSTM:
            print("CLSTM training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.CBiLSTM:
            print("CBiLSTM training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.CGRU:
            print("CGRU training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.CNN_BiLSTM:
            print("CNN_BiLSTM training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.BiGRU:
            print("BiGRU training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.CNN_BiGRU:
            print("CNN_BiGRU training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
        elif config.CNN_MUI:
            print("CNN_MUI training start......")
            model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, config)
        elif config.DEEP_CNN_MUI:
            print("DEEP_CNN_MUI training start......")
            model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, config)
        elif config.HighWay_CNN is True:
            print("HighWay_CNN training start......")
            model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, config)
        elif config.HighWay_BiLSTM_1 is True:
            print("HighWay_BiLSTM_1 training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)
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
            shutil.copy("./Test_Result.txt", "./snapshot/" + config.mulu + "/Test_Result.txt")


def main():
    """
        main function
    """
    # define word dict
    define_dict()
    # load data
    train_iter, dev_iter, test_iter = Load_Data()
    # load pretrain embedding
    load_preEmbedding()
    # update config and print
    update_arguments()
    save_arguments()
    model = load_model()
    start_train(model, train_iter, dev_iter, test_iter)


if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Neural Networks")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    config = parser.parse_args()

    config = configurable.Configurable(config_file=config.config_file)
    if config.cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())
    main()





