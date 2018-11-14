
from configparser import ConfigParser
import os


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, optionstr):
        return optionstr


class Configurable(myconf):
    def __init__(self, config_file):
        config = myconf()
        config.read(config_file)
        self._config = config

        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, ":", v)
        # if not os.path.isdir(self.save_dir):
        #     os.mkdir(self.save_dir)
        config.write(open(config_file, 'w'))

    # Data
    @property
    def word_Embedding(self):
        return self._config.getboolean('Data', 'word_Embedding')

    @property
    def freq_1_unk(self):
        return self._config.getboolean('Data', 'freq_1_unk')

    @property
    def word_Embedding_Path(self):
        return self._config.get('Data', 'word_Embedding_Path')

    @property
    def datafile_path(self):
        return self._config.get('Data', 'datafile_path')

    @property
    def name_trainfile(self):
        return self._config.get('Data', 'name_trainfile')

    @property
    def name_devfile(self):
        return self._config.get('Data', 'name_devfile')

    @property
    def name_testfile(self):
        return self._config.get('Data', 'name_testfile')

    @property
    def min_freq(self):
        return self._config.getint('Data', 'min_freq')

    @property
    def word_data(self):
        return self._config.getboolean('Data', 'word_data')

    @property
    def char_data(self):
        return self._config.getboolean('Data', 'char_data')

    @property
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

    @property
    def epochs_shuffle(self):
        return self._config.getboolean('Data', 'epochs_shuffle')

    @property
    def FIVE_CLASS_TASK(self):
        return self._config.getboolean('Data', 'FIVE_CLASS_TASK')    \

    @property
    def TWO_CLASS_TASK(self):
        return self._config.getboolean('Data', 'TWO_CLASS_TASK')

    # Save
    @property
    def snapshot(self):
        value = self._config.get('Save', 'snapshot')
        if value == "None" or value == "none":
            return None
        else:
            return value

    @property
    def predict(self):
        value = self._config.get('Save', 'predict')
        if value == "None" or value == "none":
            return None
        else:
            return value

    @property
    def test(self):
        return self._config.getboolean('Save', 'test')

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @save_dir.setter
    def save_dir(self, value):
        self._config.set('Save', 'save_dir', str(value))

    @property
    def rm_model(self):
        return self._config.getboolean('Save', 'rm_model')

    # Model
    @property
    def static(self):
        return self._config.getboolean("Model", "static")

    @property
    def wide_conv(self):
        return self._config.getboolean("Model", "wide_conv")

    @property
    def CNN(self):
        return self._config.getboolean("Model", "CNN")

    @property
    def HighWay_CNN(self):
        return self._config.getboolean("Model", "HighWay_CNN")

    @property
    def CNN_MUI(self):
        return self._config.getboolean("Model", "CNN_MUI")

    @property
    def DEEP_CNN(self):
        return self._config.getboolean("Model", "DEEP_CNN")

    @property
    def DEEP_CNN_MUI(self):
        return self._config.getboolean("Model", "DEEP_CNN_MUI")

    @property
    def LSTM(self):
        return self._config.getboolean("Model", "LSTM")

    @property
    def GRU(self):
        return self._config.getboolean("Model", "GRU")

    @property
    def BiLSTM(self):
        return self._config.getboolean("Model", "BiLSTM")

    @property
    def BiLSTM_1(self):
        return self._config.getboolean("Model", "BiLSTM_1")

    @property
    def HighWay_BiLSTM_1(self):
        return self._config.getboolean("Model", "HighWay_BiLSTM_1")

    @property
    def CNN_LSTM(self):
        return self._config.getboolean("Model", "CNN_LSTM")

    @property
    def CNN_BiLSTM(self):
        return self._config.getboolean("Model", "CNN_BiLSTM")

    @property
    def CLSTM(self):
        return self._config.getboolean("Model", "CLSTM")

    @property
    def CBiLSTM(self):
        return self._config.getboolean("Model", "CBiLSTM")

    @property
    def CGRU(self):
        return self._config.getboolean("Model", "CGRU")

    @property
    def BiGRU(self):
        return self._config.getboolean("Model", "BiGRU")

    @property
    def CNN_BiGRU(self):
        return self._config.getboolean("Model", "CNN_BiGRU")

    @property
    def embed_dim(self):
        return self._config.getint("Model", "embed_dim")

    @property
    def lstm_hidden_dim(self):
        return self._config.getint("Model", "lstm_hidden_dim")

    @property
    def lstm_num_layers(self):
        return self._config.getint("Model", "lstm_num_layers")

    @property
    def batch_normalizations(self):
        return self._config.getboolean("Model", "batch_normalizations")

    @property
    def bath_norm_momentum(self):
        return self._config.getfloat("Model", "bath_norm_momentum")

    @property
    def batch_norm_affine(self):
        return self._config.getboolean("Model", "batch_norm_affine")

    @property
    def dropout(self):
        return self._config.getfloat("Model", "dropout")

    @property
    def dropout_embed(self):
        return self._config.getfloat("Model", "dropout_embed")

    @property
    def max_norm(self):
        value = self._config.get("Model", "max_norm")
        if value == "None" or value == "none":
            return None
        else:
            return value

    @property
    def clip_max_norm(self):
        return self._config.getint("Model", "clip_max_norm")

    @property
    def kernel_num(self):
        return self._config.getint("Model", "kernel_num")

    @property
    def kernel_sizes(self):
        value = self._config.get("Model", "kernel_sizes")
        # print(list(value))
        value = [int(k) for k in list(value) if k != ","]
        return value

    @kernel_sizes.setter
    def kernel_sizes(self, value):
        self._config.set("Model", "kernel_sizes", str(value))

    @property
    def init_weight(self):
        return self._config.getboolean("Model", "init_weight")

    @property
    def init_weight_value(self):
        return self._config.getfloat("Model", "init_weight_value")

    # Optimizer
    @property
    def learning_rate(self):
        return self._config.getfloat("Optimizer", "learning_rate")

    @property
    def Adam(self):
        return self._config.getboolean("Optimizer", "Adam")

    @property
    def SGD(self):
        return self._config.getboolean("Optimizer", "SGD")

    @property
    def Adadelta(self):
        return self._config.getboolean("Optimizer", "Adadelta")

    @property
    def momentum_value(self):
        return self._config.getfloat("Optimizer", "optim_momentum_value")

    @property
    def weight_decay(self):
        return self._config.getfloat("Optimizer", "weight_decay")

    # Train
    @property
    def num_threads(self):
        return self._config.getint("Train", "num_threads")

    @property
    def device(self):
        return self._config.getint("Train", "device")

    @property
    def cuda(self):
        return self._config.getboolean("Train", "cuda")

    @property
    def epochs(self):
        return self._config.getint("Train", "epochs")

    @property
    def batch_size(self):
        return self._config.getint("Train", "batch_size")

    @property
    def log_interval(self):
        return self._config.getint("Train", "log_interval")

    @property
    def test_interval(self):
        return self._config.getint("Train", "test_interval")

    @property
    def save_interval(self):
        return self._config.getint("Train", "save_interval")




