## Introduction

* The sentiment classification tasks that apply multiple neural network
* the repository is being updated......

## Requirement

* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy

## Result

* update later......

## How to use the every folder or file

- the file of **hyperparams.py** contains all hyperparams that need to modify, based on yours nedds, select neural networks what you want and config the hyperparams.

- the file of **main-hyperparams.py** is the main function,run the command ("python main_hyperparams.py") to execute the demo.

- the folder of **models** contains all neural networks models,likes ***CNN,DeepCNN,CLSTM,CBiLSTM,CGRU,CNN_LSTM,CNN_BiLSTM,CNN_BiGRU,LSTM,BiLSTM,GRU,CNN_MUI,DeepCNN_MUI,HighWay_CNN,High_BiLSTM***.

- the file of **train_ALL_CNN.py** is the train function about CNN

- the file of **train_ALL_LSTM.py,train_ALL_LSTM_1.py** is the train function about LSTM

- the folder of **loaddata** contains some file of load dataset

- the folder of **word2vec** is the file of word embedding that you want to use

- the folder of **data** contains the dataset file,contains train data,dev data,test data.

- the file of **Parameters.txt** is being used to save all parameters values.

- the file of **Test_Result.txt** is being used to save the result of test,in the demo,save a model and test a model immediately,and int the end of training, will calculate the best result value.

## How to use the Word Embedding in demo? 

- the word embedding file saved in the folder of **word2vec**, but now is empty, because of it is to big,so if you want to use word embedding,you can to download word2vec or glove file, then saved in the folder of word2vec,and make the option of word_Embedding to True and modifiy the value of word_Embedding_Path in the **hyperparams.py** file.


## How to config hyperparams in the file of hyperparams.py

- **learning_rate**: initial learning rate.

- **epochs**:number of epochs for train

- **batch_size**：batch size for training

- **log_interval**：how many steps to wait before logging training status

- **test_interval**：how many steps to wait before testing

- **save_interval**：how many steps to wait before saving

- **save_dir**：where to save the snapshot

- **datafile_path**：datafile path

- **name_trainfile**：name of the train file

- **name_devfile**：name of the dev file

- **name_testfile**: name of the test file

- **char_data**: whether to use the strategy of char-level data

- **shuffle**:whether to shuffle the dataset when load dataset

- **epochs_shuffle**:whether to shuffle the dataset when train in every epoch

- **FIVE-CLASS-TASK**:execute five-classification-task 

- **TWO-CLASS-TASK**:execute two-classification-task 

- **dropout**:the probability for dropout

- **max_norm**:l2 constraint of parameters

- **clip-max-norm**:the values of prevent the explosion and Vanishing in Gradient

- **kernel_sizes**:comma-separated kernel size to use for convolution

- **kernel_num**:number of each kind of kernel

- **static**:whether to update the gradient during train

- **Adam**:select the optimizer of adam

- **SGD**：select the optimizer of SGD

- **Adadelta**:select the optimizer of Adadelta

- **optim-momentum-value**:the parameter in the optimizer

- **wide_conv**:whether to use wide convcolution True : wide  False : narrow

- **batch_normalizations**:whether to use batch normalizations in the model

- **bath_norm_momentum**:the parameter value of batch_normalizations

- **batch_norm_affine**:the parameter value of batch_normalizations

- **min_freq**:min freq to include during built the vocab when use torchtext, default is 1

- **word_Embedding**: use word embedding

- **embed_dim**:number of embedding dimension

- **word-Embedding-Path**:the path of word embedding file

- **lstm-hidden-dim**:the hidden dim with lstm model

- **lstm-num-layers**:the num of hidden layers with lstm

- **no_cuda**: no use cuda

- **num_threads**:set the value of threads when run the demo

- **init_weight**:whether to init weight

- **init-weight-value**:the value of init weight

- **weight-decay**:L2 weight_decay,default value is zero in optimizer

- **seed_num**:set the num of random seed

- **rm_model**:whether to delete the model after test acc so that to save space


## Reference 

- [http://www.cnblogs.com/bamtercelboo/p/7469005.html](http://www.cnblogs.com/bamtercelboo/p/7469005.html "基于pytorch的CNN-LSTM神经网络模型调参小结")

- later update
