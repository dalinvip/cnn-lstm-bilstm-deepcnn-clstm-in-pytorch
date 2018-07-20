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
from DataLoader.load_external_word_embedding import Word_Embedding
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
from DataUtils.Common import seed_num
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


def main():
    """
        main function
    """
    # define word dict
    define_dict()

    # load data
    train_iter, dev_iter, test_iter = Load_Data()


if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Neural Networks")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    config = parser.parse_args()

    config = configurable.Configurable(config_file=config.config_file)
    if config.no_cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
    print("torch.cuda.initial_seed", torch.cuda.initial_seed())
    main()





