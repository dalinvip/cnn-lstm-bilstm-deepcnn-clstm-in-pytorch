## Chinese-NER Task Config ##

- Use `ConfigParser` to config parameter  
	- `from configparser import ConfigParser`  .
	- Detail see `config.py` and `config.cfg`, please.  

- Following is `config.cfg` Parameter details.

- [Data]
	- `pretrained_embed` (True or False) ------ whether to use pretrained embedding.

	- ` pretrained-embed-file` (path)  ------ word embedding file path(`Pretrain_Embedding`).

	- `train-file/dev-file/test-file`(path)  ------ train/dev/test data path(`Data`).

	- `min_freq` (integer number) ------ The smallest Word frequency when build vocab.

	- `shuffle/epochs-shuffle`(True or False) ------ shuffle data .

- [Save]
	- `save_direction` (path) ------ save model path.

	- `rm_model` (True or False) ------ remove model to save space(now not use).

- [Model]

	- `model-bilstm` (True or False) ------ Bilstm model.

	- `lstm-layers` (integer) ------ number layers of lstm.

	- `embed-dim` (integer) ------ embedding dim = pre-trained embedding dim.

	- `embed-finetune` (True or False) ------ word embedding finetune or no-finetune.

	- `lstm-hiddens` (integer) ------numbers of lstm hidden.

	- `dropout-emb/dropout `(float) ------ dropout for prevent overfitting.

	- `windows-size` (integer) ------ Context window feature size.

- [Optimizer]

	- `adam` (True or False) ------ `torch.optim.Adam`

	- `sgd` (True or False)  ------ `torch.optim.SGD`

	- `learning-rate`(float) ------ learning rate(0.001, 0.01).

	- ` learning-rate-decay`(float) ------ learning rate decay.

	- `weight-decay` (float) ------ 1e-8.

	- `clip-max-norm` (Integer number) ------ 5, 10, 15.

- [Train]

	- `num-threads` (Integer) ------ threads.

	- `use-cuda` (True or False) ------ support `cuda` speed up.

	- `epochs` (Integer) ------ train epochs

	- `batch-size/dev-batch-size/test-batch-size` (Integer) ------ number of batch

	- `log-interval`(Integer) ------ steps of print log.