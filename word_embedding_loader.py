import torch
import numpy as np
import random
torch.manual_seed(233)
random.seed(233)

def vector_loader(text_field_words):
    # load word2vec_raw
    # path = 'word_embedding/glove.6B.300d.txt'
    path = 'word2vec/glove.sentiment.conj.pretrained.txt'
    words = []
    words_dict = {}
    file = open(path, 'rt', encoding='utf-8')
    lines = file.readlines()
    t = 300

    for line in lines:
        line_split = line.split(' ')
        word = line_split[0]
        nums = line_split[1:]
        nums = [float(e) for e in nums]
        # data.append(line_list)
        words.append(word)
        words_dict[word] = nums

    # match
    count_list2 = []
    count = 0
    dict_cat = []
    for word in text_field_words:
        if word in words_dict:
            count += 1
            dict_cat.append(words_dict[word])
        else:
            dict_cat.append([0.0] * t)
            count += 1
            count_list2.append(count - 1)
    count_data = len(text_field_words) - len(count_list2)

    # modify zero
    sum = []
    for j in range(t):
        sum_col = 0.0
        for i in range(len(dict_cat)):
            sum_col += dict_cat[i][j]
            sum_col = float(sum_col / count_data)
            sum_col = round(sum_col, 6)
        sum.append(sum_col)
    print("sum ",sum)
    # sum_none = []
    # for i in range(t):
    #     sum_total = sum[i] / (len(dict_cat) - len(count_list2))
    #     sum_total = round(sum_total, 6)
    #     sum_none.append(sum_total)
    # # print(sum_none)

    for i in range(len(count_list2)):
        dict_cat[count_list2[i]] = sum

    return dict_cat


def vector_loader_zero(text_field_words):
    # load word2vec_raw
    path = 'word_embedding/glove.6B.300d.txt'
    words = []
    words_dict = {}
    file = open(path, 'rt', encoding='utf-8')
    lines = file.readlines()
    t = 300

    for line in lines:
        line_split = line.split(' ')
        word = line_split[0]
        nums = line_split[1:]
        nums = [float(e) for e in nums]
        # data.append(line_list)
        words.append(word)
        words_dict[word] = nums

    # match
    count_list2 = []
    count = 0
    dict_cat = []
    for word in text_field_words:
        if word in words_dict:
            count += 1
            dict_cat.append(words_dict[word])
        else:
            dict_cat.append([0.0] * t)
            # count += 1
            # count_list2.append(count - 1)

    # # modify zero
    # sum = []
    # for j in range(t):
    #     sum_col = 0.0
    #     for i in range(len(dict_cat)):
    #         sum_col += dict_cat[i][j]
    #         sum_col = round(sum_col, 6)
    #     # sum.append(sum_col)
    #
    # sum_none = []
    # for i in range(t):
    #     sum_total = sum[i] / (len(sum) - len(count_list2))
    #     sum_total = round(sum_total, 6)
    #     sum_none.append(sum_total)
    # # print(sum_none)
    #
    # for i in range(len(count_list2)):
    #     dict_cat[count_list2[i]] = sum_none

    return dict_cat


def vector_loader_modify(text_field_words):
    # load word2vec_raw
    path = 'word_embedding/glove.6B.300d.txt'
    words = []
    words_dict = {}
    file = open(path, 'rt', encoding='utf-8')
    lines = file.readlines()
    t = 300

    for line in lines:
        line_split = line.split(' ')
        word = line_split[0]
        nums = line_split[1:]
        nums = [float(e) for e in nums]
        # data.append(line_list)
        words.append(word)
        words_dict[word] = nums


    uniform = np.random.uniform(-0.1, 0.1, t).round(6).tolist()     # uniform distribution U(a,b).均匀分布
    # match
    count_list2 = []
    count = 0
    dict_cat = []
    for word in text_field_words:
        if word in words_dict:
            count += 1
            dict_cat.append(words_dict[word])
        else:
            # a = torch.normal(mean=0.0, std=torch.arange(0.09, 0, -0.09))
            dict_cat.append(uniform)
            count += 1
            count_list2.append(count - 1)
    # count_data = len(text_field_words) - len(count_list2)

    # # modify uniform
    # sum = []
    # for j in range(t):
    #     sum_col = 0.0
    #     for i in range(len(dict_cat)):
    #         sum_col += dict_cat[i][j]
    #         sum_col = float(sum_col / count_data)
    #         sum_col = round(sum_col, 6)
    #     sum.append(sum_col)

    # sum_none = []
    # for i in range(t):
    #     sum_total = sum[i] / (len(sum) - len(count_list2))
    #     sum_total = round(sum_total, 6)
    #     sum_none.append(sum_total)
    # # print(sum_none)
    #
    # for i in range(len(count_list2)):
    #     dict_cat[count_list2[i]] = sum_none

    return dict_cat


import torch
import numpy
def vector_loader_rand(text_field_words):
    t = 300
    # match
    text_words_size = len(text_field_words)
    dict_cat = torch.randn(text_words_size, t)
    dict_cat = dict_cat.numpy()
    dict_cat = dict_cat.tolist()

    return dict_cat

# {'ü', 'q', 'ó', 'á', '=', 'l', 'â', ':', 'i', 'ö', 'à', ',', '(', '4', 'û', 'b', 'n', 'e', 's', '`', "'", 'm', '1', 'c', '\\', '/', '.', 'h', ';', '&', 'f', '%', '$', 'ï', 'u', 'a', 'v', 'o', 'z', '#', 'ã', 'y', '0', '2', '7', '5', 'j', '?', '<unk>', '-', '<pad>', '*', '!', 'w', 'd', 'p', 'è', 'í', 'k', '8', 'é', '+', ')', 'r', '9', '3', 'ñ', 'æ', 'g', 'x', '6', 't'}
