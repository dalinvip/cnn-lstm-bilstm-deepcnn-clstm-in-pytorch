# coding=utf-8
import numpy as np
import collections
import random
from DataUtils.Common import seed_num
# import hyperparams
np.random.seed(seed_num)
random.seed(seed_num)


class Word_Embedding():
    def __init__(self):
        print("loading external word embedding")
        self.test = 0

    def test(self):
        print(self.test)

    # load word embedding
    def load_my_vecs(self, path, vocab, freqs, k=None):
        word_vecs = collections.OrderedDict()
        with open(path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                count += 1
                if word in vocab:  # whether to judge if in vocab
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        if count <= k:
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    # solve unknown by avg word embedding
    def add_unknown_words_by_avg(self, word_vecs, vocab, k=100):
        # solve unknown words inplaced by zero list
        word_vecs_numpy = []
        for word in vocab:
            if word in word_vecs:
                word_vecs_numpy.append(word_vecs[word])
        print(len(word_vecs_numpy))
        col = []
        for i in range(k):
            sum = 0.0
            for j in range(int(len(word_vecs_numpy))):
                sum += word_vecs_numpy[j][i]
                sum = round(sum, 6)
            col.append(sum)
        zero = []
        for m in range(k):
            avg = col[m] / (len(word_vecs_numpy))
            avg = round(avg, 6)
            zero.append(float(avg))

        list_word2vec = []
        oov = 0
        iov = 0
        for word in vocab:
            if word not in word_vecs:
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
    def add_unknown_words_by_uniform(self, word_vecs, vocab, k=100):
        list_word2vec = []
        oov = 0
        iov = 0
        for word in vocab:
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        print("oov count", oov)
        print("iov count", iov)
        return list_word2vec

    # load word embedding
    def load_my_vecs_freq1(self, path, vocab, freqs, pro):
        # word_vecs = {}
        word_vecs = collections.OrderedDict()
        with open(path, encoding="utf-8") as f:
            freq = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                if word in vocab:  # whehter to judge if in vocab
                    if freqs[word] == 1:
                        a = np.random.uniform(0, 1, 1).round(2)
                        if pro < a:
                            continue
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs
