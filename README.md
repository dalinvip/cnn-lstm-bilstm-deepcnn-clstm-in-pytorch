## Introduction
* Themotional classification tasks that apply multiple neural network
* the repository is being updated......
## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy

## Result
* update later......

## Usage


- python3 main.py -h

- CNN text classificer

- optional arguments:
 

-  -h, --help            -------->  show this help message and exit

-  -lr LR                -------->initial learning rate [default: 0.001]

-  -epochs EPOCHS        -------->number of epochs for train [default: 256]

-  -batch-size BATCH_SIZE -------->batch size for training [default: 64]

-  -log-interval LOG_INTERVAL -------->how many steps to wait before logging training status[default: 1]

-  -test-interval TEST_INTERVAL -------->how many steps to wait before testing [default: 100]

-  -save-interval SAVE_INTERVAL -------->how many steps to wait before saving [default:500]

-  -save-dir SAVE_DIR    -------->where to save the snapshot

-  -shuffle              -------->shuffle the data every epoch

-  -dropout DROPOUT      -------->the probability for dropout [default: 0.5]

-  -max-norm MAX_NORM    -------->l2 constraint of parameters [default: 3.0]

-  -embed-dim EMBED_DIM  -------->number of embedding dimension [default: 128]

-  -kernel-num KERNEL_NUM -------->number of each kind of kernel

-  -kernel-sizes KERNEL_SIZES  -------->comma-separated kernel size to use for convolution
  
-  -static               -------->fix the embedding
  
-  -FIVE_CLASS_TASK      -------->whether to execute five-classification-task
 
-  -TWO_CLASS_TASK       -------->whether to execute two-classification-task
  
-  -CNN                  -------->whether to use CNN model
  
-  -DEEP_CNN             -------->whether to use Depp CNN model 

-  -LSTM                 -------->whether to use LSTM model
  
-  -BiLSTM               -------->whether to use Bi-LSTM model
 
-  -BiLSTM_1             -------->whether to use Bi-LSTM_1 model
 
-  -CNN_LSTM             -------->whether to use CNN_LSTM model
  
-  -CLSTM                -------->whether to use CLSTM model  
 
-  -word_Embedding       -------->whether to load word embedding
  
-  -lstm-hidden-dim LSTM_HIDDEN_DIM-------->the number of embedding dimension in LSTM hidden layer
  
-  -device DEVICE        -------->device to use for iterate data, -1 mean cpu [default:-1]
 
-  -no_cuda              -------->disable the gpu
  
-  -snapshot SNAPSHOT    -------->filename of model snapshot [default: None]
 
-  -predict PREDICT      -------->predict the sentence given
 
-  -test                 -------->train or test

## Word Embedding Memo 
- the word embedding saved in the folder of word2vec, but now is empty, because of it is to big,so if you want to use word embedding,you can to download word2vce or glove, then saved in the folder of word2vec,and make the option of word_Embedding to True and modifiy the name in the demo.
- in the demo,I used following......
    `if args.embed_dim == 100:
        path = "./word2vec/glove.6B.100d.txt"
    elif args.embed_dim == 200:
        path = "./word2vec/glove.6B.200d.txt"
    elif args.embed_dim == 300:
        path = "./word2vec/glove.6B.300d.txt"`


## Neural Networks 
-  -CNN                  -------->single layer CNN model
  
-  -DEEP_CNN             -------->double layer CNN model 

-  -LSTM                 -------->LSTM model
  
-  -BiLSTM               -------->Bi-LSTM model is a kind of bidirection LSTM
 
-  -BiLSTM_1             -------->Bi-LSTM_1 model is the anthoer way of the Bi-LSTM
 
-  -CNN_LSTM             -------->after the pooling of CNN and LSTM, CNN and LSTM to combine to linear
  
-  -CLSTM                -------->the output of the CNN as the LSTM input 


## File Memo
- Parameters.txt  --------> the file is being used to save all parameters values.
- Test_Result.txt --------> the file is being used to save the result of test,in the demo,save a model and test a model immediately,and int the end of training, will calculate the best result value.
## Reference 

- later update

