# @Author : bamtercelboo
# @Datetime : 2018/07/10 16.03
# @File : train.py
# @Last Modify Time : 2018/07/10 16.03
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Load_Pretrained_Embed.py
    FUNCTION : loading pretrained word embedding
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
import numpy as np
import tqdm

from DataUtils.Common import *
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def load_pretrained_emb_zeros(path, text_field_words_dict, pad=None, set_padding=False):
    print("loading pre_train embedding by zeros......")
    if not isinstance(text_field_words_dict, dict):
        text_field_words_dict = convert_list2dict(text_field_words_dict)
    if pad is not None:
        padID = text_field_words_dict[pad]
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    f.close()
    word_count = len(text_field_words_dict)
    print('The number of wordsDict is {} \nThe dim of pretrained embedding is {}'.format(str(word_count),
                                                                                           str(embedding_dim)))
    embeddings = np.zeros((int(word_count), int(embedding_dim)))
    iv_num = 0
    oov_num = 0
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        # lines = tqdm.tqdm(lines)
        for line in lines:
            values = line.strip().split(' ')
            if len(values) == 1 or len(values) == 2:
                continue
            index = text_field_words_dict.get(values[0])  # digit or None
            if index:
                iv_num += 1
                vector = np.array([float(i) for i in values[1:]], dtype='float32')
                embeddings[index] = vector

    f.close()
    oov_num = word_count - iv_num
    print("iv_num {} oov_num {} oov_radio {:.4f}%".format(iv_num, oov_num, round((oov_num / word_count) * 100, 4)))
    return torch.from_numpy(embeddings).float()


def load_pretrained_emb_Embedding(path, text_field_words_dict, pad=None, set_padding=False):
    print("loading pre_train embedding by nn.Embedding......")
    if not isinstance(text_field_words_dict, dict):
        text_field_words_dict = convert_list2dict(text_field_words_dict)
    if pad is not None:
        padID = text_field_words_dict[pad]
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    f.close()
    word_count = len(text_field_words_dict)
    print('The number of wordsDict is {} \nThe dim of pretrained embedding is {}'.format(str(word_count),
                                                                                         str(embedding_dim)))
    embed = nn.Embedding(int(word_count), int(embedding_dim))
    init.xavier_uniform(embed.weight.data)
    embeddings = np.array(embed.weight.data)
    iv_num = 0
    oov_num = 0
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        lines = tqdm.tqdm(lines)
        for line in lines:
            values = line.strip().split(' ')
            if len(values) == 1 or len(values) == 2:
                continue
            index = text_field_words_dict.get(values[0])  # digit or None
            if index:
                iv_num += 1
                vector = np.array([float(i) for i in values[1:]], dtype='float32')
                embeddings[index] = vector

    f.close()
    oov_num = word_count - iv_num
    print("iv_num {} oov_num {} oov_radio {:.4f}%".format(iv_num, oov_num, round((oov_num / word_count) * 100, 4)))
    return torch.from_numpy(embeddings).float()


def load_pretrained_emb_avg(path, text_field_words_dict, pad=None, set_padding=False):
    print("loading pre_train embedding by avg......")
    if not isinstance(text_field_words_dict, dict):
        text_field_words_dict = convert_list2dict(text_field_words_dict)
    assert pad is not None, "pad not allow with None"
    padID = text_field_words_dict[pad]
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    f.close()
    word_count = len(text_field_words_dict)
    print('The number of wordsDict is {} \nThe dim of pretrained embedding is {}\n'.format(str(word_count),
                                                                                           str(embedding_dim)))
    embeddings = np.zeros((int(word_count), int(embedding_dim)))

    inword_list = {}
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        lines = tqdm.tqdm(lines)
        for line in lines:
            lines.set_description("Processing")
            values = line.strip().split(" ")
            if len(values) == 1 or len(values) == 2:
                continue
            index = text_field_words_dict.get(values[0])  # digit or None
            if index:
                vector = np.array([float(i) for i in values[1:]], dtype='float32')
                embeddings[index] = vector
                inword_list[index] = 1
    f.close()
    print("oov words initial by avg embedding, maybe take a while......")
    sum_col = np.sum(embeddings, axis=0) / len(inword_list)     # avg
    for i in range(len(text_field_words_dict)):
        if i not in inword_list and i != padID:
            embeddings[i] = sum_col

    OOVWords = word_count - len(inword_list)
    oov_radio = np.round(OOVWords / word_count, 6)
    print("All Words = {}, InWords = {}, OOVWords = {}, OOV Radio={}".format(
        word_count, len(inword_list), OOVWords, oov_radio))

    return torch.from_numpy(embeddings).float()


def load_pretrained_emb_uniform(path, text_field_words_dict, pad=None, set_padding=False):
    print("loading pre_train embedding by uniform......")
    if not isinstance(text_field_words_dict, dict):
        text_field_words_dict = convert_list2dict(text_field_words_dict)
    assert pad is not None, "pad not allow with None"
    padID = text_field_words_dict[pad]
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    f.close()
    word_count = len(text_field_words_dict)
    print('The number of wordsDict is {} \nThe dim of pretrained embedding is {}\n'.format(str(word_count),
                                                                                           str(embedding_dim)))
    embeddings = np.zeros((int(word_count), int(embedding_dim)))

    inword_list = {}
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        lines = tqdm.tqdm(lines)
        for line in lines:
            lines.set_description("Processing")
            values = line.strip().split(" ")
            if len(values) == 1 or len(values) == 2:
                continue
            index = text_field_words_dict.get(values[0])  # digit or None
            if index:
                vector = np.array([float(i) for i in values[1:]], dtype='float32')
                embeddings[index] = vector
                inword_list[index] = 1
    f.close()
    print("oov words initial by uniform embedding, maybe take a while......")
    # sum_col = np.sum(embeddings, axis=0) / len(inword_list)     # avg
    uniform_col = np.random.uniform(-0.25, 0.25, int(embedding_dim)).round(6)    # avg
    for i in range(len(text_field_words_dict)):
        if i not in inword_list and i != padID:
            embeddings[i] = uniform_col

    OOVWords = word_count - len(inword_list)
    oov_radio = np.round(OOVWords / word_count, 6)
    print("All Words = {}, InWords = {}, OOVWords = {}, OOV Radio={}".format(
        word_count, len(inword_list), OOVWords, oov_radio))

    return torch.from_numpy(embeddings).float()


def convert_list2dict(convert_list):
    list_dict = OrderedDict()
    for index, word in enumerate(convert_list):
        list_dict[word] = index
    return list_dict


