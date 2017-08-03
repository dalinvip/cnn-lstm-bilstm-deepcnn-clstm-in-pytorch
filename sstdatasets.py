from nltk.tree import Tree
import os

from torchtext import data as data


class SST(data.ZipDataset):

    # url = 'http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    # filename = 'trainDevTestTrees_PTB.zip'
    # dirname = 'ssttrees'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, subtrees=False, examples=None,
                 fine_grained=False, **kwargs):
        """Create an SST dataset instance given a path and fields.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]

        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[label]
        # label_field.preprocessing = data.Pipeline(get_label_str)
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(path) as f:
                for line in f:
                    if line[-2] == '0':
                        examples += [data.Example.fromlist([line[:line.find('|')], 'negative'], fields=fields)]
                    elif line[-2] == '1':
                        examples += [data.Example.fromlist([line[:line.find('|')], 'negative'], fields=fields)]
                    elif line[-2] == '2':
                        continue
                        # examples += [data.Example.fromlist([line[:line.find('|')], 'neutral'], fields=fields)]
                    elif line[-2] == '3':
                        examples += [data.Example.fromlist([line[:line.find('|')], 'positive'], fields=fields)]
                    else:
                        examples += [data.Example.fromlist([line[:line.find('|')], 'positive'], fields=fields)]
        super(SST, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.',
               train='raw.clean.train', validation='raw.clean.dev', test='raw.clean.test',
               train_subtrees=False, **kwargs):
        """Create dataset objects for splits of the SSTB dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.txt'.
            train_subtrees: Whether to use all subtrees in the training set.
                Default: False.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # path = cls.download_or_unzip(root)

        path = "./data/"
        print(path + train)
        print(path + validation)
        print(path + test)

        train_data = None if train is None else cls(
            path + train, text_field, label_field, subtrees=train_subtrees,
            **kwargs)
        val_data = None if validation is None else cls(
            path + validation, text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            path + test, text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.', wv_dir='.',
              wv_type=None, wv_dim='300d', **kwargs):
        """Creater iterator objects for splits of the SSTB dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, wv_dir=wv_dir, wv_type=wv_type, wv_dim=wv_dim)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)
