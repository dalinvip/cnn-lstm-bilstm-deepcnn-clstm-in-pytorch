# coding=utf-8
# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : mydatasets_self_five.py
# @Last Modify Time : 2018/07/19 22:35

import re
import os
import tarfile
from six.moves import urllib
from torchtext import data
import random
import torch
from DataUtils.Common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)


class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tfile, root)
        return os.path.join(path, '')


class MR(TarDataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, file=None, examples=None, char_data=None, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)

            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = None if os.path.join(path, file) is None else os.path.join(path, file)
            examples = []
            with open(path) as f:
                a, b, c, d, e = 0, 0, 0, 0, 0
                for line in f:
                    sentence, flag = line.strip().split(' ||| ')
                    if char_data is True:
                        sentence = sentence.split(" ")
                        sentence = MR.char_data(self, sentence)
                    # print(sentence)
                    # clear string in every sentence
                    sentence = clean_str(sentence)
                    if line[-2] == '0':
                        a += 1
                        examples += [data.Example.fromlist([sentence, "very negative"], fields=fields)]
                    elif line[-2] == '1':
                        b += 1
                        examples += [data.Example.fromlist([sentence, 'negative'], fields=fields)]
                    elif line[-2] == '2':
                        c += 1
                        examples += [data.Example.fromlist([sentence, 'neutral'], fields=fields)]
                    elif line[-2] == '3':
                        d += 1
                        examples += [data.Example.fromlist([sentence, 'positive'], fields=fields)]
                    else:
                        e += 1
                        examples += [data.Example.fromlist([sentence, 'very positive'], fields=fields)]
                print("a {} b {} c {} d {} e {}".format(a, b, c, d, e))
        super(MR, self).__init__(examples, fields, **kwargs)

    def char_data(self, list):
        data = []
        for i in range(len(list)):
            for j in range(len(list[i])):
                data += list[i][j]
        return data


    @classmethod
    def splits(cls, path, train, dev, test, char_data, text_field, label_field, dev_ratio=.1, shuffle=True ,root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # path = cls.download_or_unzip(root)
        # path = "./data/"
        print(path + train)
        print(path + dev)
        print(path + test)
        examples_train = cls(text_field, label_field, path=path, file=train, char_data=char_data, **kwargs).examples
        examples_dev = cls(text_field, label_field, path=path, file=dev, char_data=char_data, **kwargs).examples
        examples_test = cls(text_field, label_field, path=path, file=test,char_data=char_data, **kwargs).examples
        if shuffle:
            print("shuffle data examples......")
            random.shuffle(examples_train)
            random.shuffle(examples_dev)
            random.shuffle(examples_test)

        return (cls(text_field, label_field, examples=examples_train),
                cls(text_field, label_field, examples=examples_dev),
                cls(text_field, label_field, examples=examples_test))
