# coding=utf-8
# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : handle_wordEmbedding2File.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    handle external word embedding to file
"""
import os
import tqdm


class WordEmbedding2File:
    def __init__(self, wordEmbedding_path, data_path, extract_path):
        print("handling external word embedding to file")
        self.wordEmbedding_path = wordEmbedding_path
        self.data_path = data_path
        self.extract_path = extract_path
        self.data_dict = self.read_data(data_path)
        self.extract_dict = {}
        self.dim = 100
        self.read_vectors(self.wordEmbedding_path)
        self.write(self.extract_path, self.extract_dict)
        # print(self.data_dict)
        # print(self.extract_dict)

    def read_data(self, path):
        print("read data file {}".format(path))
        data_list = []
        with open(path, encoding="UTF-8") as f:
            for line in f.readlines():
                line = line.strip("\n").split(" ")[:-2]
                data_list.extend(line)
        return set(data_list)

    def read_vectors(self, path):
        print("read embedding path {}".format(path))
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            self.dim = len(lines[2].strip("\n").strip().split(" ")[1:-1])
            # print(dim)
            lines = tqdm.tqdm(lines)
            for line in lines:
                values = line.strip("\n").strip().split(" ")
                if len(values) == 1 or len(values) == 2 or len(values) == 3:
                    continue
                word, vector = values[0], values[1:-1]
                if word in self.data_dict:
                    self.extract_dict[word] = vector

    def write(self, path, dict):
        print("writing to {}".format(path))
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        all_words, dim = len(dict), self.dim
        print(all_words, dim)
        file.write(str(all_words) + " " + str(dim) + "\n")
        for word in dict:
            value = " ".join(dict[word])
            v = word + " " + value + "\n"
            file.write(v)


if __name__ == "__main__":
    wordEmbedding_path = "GoogleNews_wordEmbedding/vectors.utf-8"
    data_path = "./sst_all.txt"
    extract_path = "./extract_googleNews_embed_sst.txt"
    WordEmbedding2File(wordEmbedding_path=wordEmbedding_path, data_path=data_path, extract_path=extract_path)

