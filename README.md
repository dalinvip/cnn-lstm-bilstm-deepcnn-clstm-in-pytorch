## Introduction ##

- A classification task implement in pytorch, contains some neural networks in [models](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch/tree/master/models).

* Recenely,  I've released the code. 
	* **old-version-17** release [here](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch/releases/tag/pytorch0.3.1-old_version_17)  
	* **pytorch version == 0.3.1** release on [here](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch/releases/tag/pytorch0.3.1)  

- This is a version of my own architecture  ---  [pytorch-text-classification](https://github.com/bamtercelboo/pytorch_text_classification)  

- **BERT For Text Classification**  --- [PyTorch_Bert_Text_Classification](https://github.com/bamtercelboo/PyTorch_Bert_Text_Classification)  


## Requirement ##

	pyorch : 1.0.1
	python : 3.6
	torchtext: 0.2.1
	cuda : 8.0 (support cuda speed up, can chose, default True)

## Usage ##
 
modify the config file, see the Config directory([here](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch/tree/master/Config)) for detail.  

	1、python main.py
	2、python main.py --config_file ./Config/config.cfg 
	3、sh run.sh

## Model ##

Contains some neural networks implement in pytorch, see the [models](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch/tree/master/models) for detail.

## Data ##

SST-1 and SST-2.

## Result ##

I haven't adjusted the hyper-parameters seriously, you can also see train log in [here](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch/tree/master/result).  

The following test set accuracy are based on the best dev set accuracy.    

| Data/Model | % SST-1 | % SST-2 |  
| ------------ | ------------ | ------------ |  
| CNN | 46.1086 | 84.2943 |  
| Bi-LSTM | 47.9186 | 86.3262 |  
| Bi-GRU | 47.6923 | 86.7655 |  


## Reference ##

- [基于pytorch的CNN-LSTM神经网络模型调参小结](http://www.cnblogs.com/bamtercelboo/p/7469005.html "基于pytorch的CNN-LSTM神经网络模型调参小结")
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
-  [Context-Sensitive Lexicon Features for Neural Sentiment Analysis](https://arxiv.org/pdf/1408.5882.pdf)

## Question ##

- if you have any question, you can open a issue or email **bamtercelboo@{gmail.com, 163.com}**.

- if you have any good suggestions, you can PR or email me.
