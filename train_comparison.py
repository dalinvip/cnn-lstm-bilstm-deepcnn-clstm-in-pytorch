# coding=utf-8
# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : train_comparison.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
统一对比训练脚本：BiLSTM、Transformer、CNN+Transformer、CNN+Transformer_Parallel
在同一数据划分和统一训练配置下进行公平对比
"""

import os
import sys
import argparse
import datetime
import json
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import torchtext.data as data
import shutil
import random
import numpy as np
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Config.config as configurable
from models.model_BiLSTM import BiLSTM
from models.model_Transformer import Transformer
from models.model_CNN_Transformer import CNN_Transformer
from models.model_CNN_Transformer_Parallel import CNN_Transformer_Parallel
from DataLoader import mydatasets_self_two
from DataUtils.Load_Pretrained_Embed import load_pretrained_emb_zeros, load_pretrained_emb_avg
from DataUtils.Common import seed_num, pad, unk

# 设置随机种子
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_num)


class BestResult:
    """记录最佳结果"""
    def __init__(self):
        self.best_dev_accuracy = -1
        self.best_test_accuracy = -1
        self.best_epoch = 1
        self.best_test = False
        self.train_time = 0


class ModelComparison:
    """模型对比类"""
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        print("\n" + "="*60)
        print("Loading data...")
        print("="*60)
        
        # 定义字段 - 注意：BiLSTM模型期望batch_first=False的输入格式
        self.text_field = data.Field(lower=True, batch_first=False, fix_length=self.args.max_len)
        self.label_field = data.Field(sequential=False, unk_token=None)
        
        # 加载数据集
        # 配置文件中的name_trainfile等已经包含路径前缀，需要提取目录和文件名
        train_path = self.config.name_trainfile
        dev_path = self.config.name_devfile
        test_path = self.config.name_testfile
        
        # 提取目录路径（去掉文件名）
        data_dir = os.path.dirname(train_path) if os.path.dirname(train_path) else './Data'
        train_file = os.path.basename(train_path)
        dev_file = os.path.basename(dev_path)
        test_file = os.path.basename(test_path)
        
        print(f"Data directory: {data_dir}")
        print(f"Train file: {train_file}")
        print(f"Dev file: {dev_file}")
        print(f"Test file: {test_file}")
        
        train_data, dev_data, test_data = mydatasets_self_two.MR.splits(
            path=data_dir,
            train=train_file,
            dev=dev_file,
            test=test_file,
            char_data=None,
            text_field=self.text_field,
            label_field=self.label_field
        )
        
        print(f"Train samples: {len(train_data)}")
        print(f"Dev samples: {len(dev_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # 构建词汇表
        self.text_field.build_vocab(train_data, min_freq=self.config.min_freq)
        self.label_field.build_vocab(train_data)
        
        print(f"Vocab size: {len(self.text_field.vocab)}")
        print(f"Classes: {self.label_field.vocab.itos}")
        
        # 创建迭代器
        self.train_iter, self.dev_iter, self.test_iter = data.Iterator.splits(
            (train_data, dev_data, test_data),
            batch_sizes=(self.args.batch_size, len(dev_data), len(test_data)),
            device=-1,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False,
            repeat=False
        )
        
        # 加载预训练词向量
        self.args.embed_num = len(self.text_field.vocab)
        # 二分类任务中，label vocab 包含 ['positive', 'negative']，没有 <unk>
        self.args.class_num = len(self.label_field.vocab)
        self.args.paddingId = self.text_field.vocab.stoi[pad]
        
        if self.config.word_Embedding:
            self.args.pretrained_weight = load_pretrained_emb_zeros(
                path=self.config.word_Embedding_Path,
                text_field_words_dict=self.text_field.vocab.itos,
                pad=pad
            )
            self.args.word_Embedding = True
        else:
            self.args.word_Embedding = False
            self.args.pretrained_weight = None
            
        print(f"Embedding dim: {self.args.embed_dim}")
        print(f"Number of classes: {self.args.class_num}")
        
    def train_model(self, model, model_name):
        """训练单个模型（带早停机制）"""
        print("\n" + "="*60)
        print(f"Training {model_name}...")
        print("="*60)
        
        if self.args.cuda:
            model.cuda()
        
        # 优化器
        if self.args.Adam:
            print("Using Adam optimizer")
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.SGD:
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=0.9)
        else:
            print("Using Adadelta optimizer")
            optimizer = torch.optim.Adadelta(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        best_result = BestResult()
        steps = 0
        start_time = time.time()
        
        # 早停参数
        patience = self.args.patience if hasattr(self.args, 'patience') else 5
        patience_counter = 0
        
        model.train()
        for epoch in range(1, self.args.epochs + 1):
            steps = 0
            epoch_loss = 0
            epoch_corrects = 0
            epoch_samples = 0
            
            for batch in self.train_iter:
                feature, target = batch.text, batch.label
                target = target.data
                
                if self.args.cuda:
                    feature, target = feature.cuda(), target.cuda()
                
                optimizer.zero_grad()
                logit = model(feature)
                loss = F.cross_entropy(logit, target)
                loss.backward()
                
                if self.args.clip_max_norm is not None:
                    utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip_max_norm)
                
                optimizer.step()
                
                steps += 1
                epoch_loss += loss.item()
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                epoch_corrects += corrects
                epoch_samples += batch.batch_size
                
                # 记录训练过程
                if steps % self.args.log_interval == 0:
                    train_acc = float(corrects) / batch.batch_size * 100.0
                    sys.stdout.write(
                        f'\rEpoch[{epoch}/{self.args.epochs}] Batch[{steps}] - '
                        f'loss: {loss.item():.6f}  acc: {train_acc:.4f}%'
                    )
            
            # 每个epoch结束后的评估
            dev_acc = self.evaluate(self.dev_iter, model)
            test_acc = self.evaluate(self.test_iter, model)
            train_acc = float(epoch_corrects) / epoch_samples * 100.0
            
            improved = False
            if dev_acc > best_result.best_dev_accuracy:
                best_result.best_dev_accuracy = dev_acc
                best_result.best_test_accuracy = test_acc
                best_result.best_epoch = epoch
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1
            
            print(f'\nEpoch[{epoch}/{self.args.epochs}] Summary:')
            print(f'  Train Loss: {epoch_loss/steps:.6f}  Train Acc: {train_acc:.4f}%')
            print(f'  Dev Acc: {dev_acc:.4f}%  Test Acc: {test_acc:.4f}%')
            print(f'  Best Dev: {best_result.best_dev_accuracy:.4f}%  Best Test: {best_result.best_test_accuracy:.4f}% (Epoch {best_result.best_epoch})')
            
            if improved:
                print(f'  *** New best model! ***')
            else:
                print(f'  Patience: {patience_counter}/{patience}')
            
            # 早停检查
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch} epochs')
                break
        
        best_result.train_time = time.time() - start_time
        return best_result
    
    def evaluate(self, data_iter, model):
        """评估模型"""
        model.eval()
        corrects = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_iter:
                feature, target = batch.text, batch.label
                target = target.data
                
                if self.args.cuda:
                    feature, target = feature.cuda(), target.cuda()
                
                logit = model(feature)
                corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                total += target.size(0)
        
        accuracy = float(corrects) / total * 100.0
        model.train()
        return accuracy
    
    def run_comparison(self):
        """运行对比实验"""
        # 加载数据
        self.load_data()
        
        # 定义要对比的模型
        models_to_compare = {
            'BiLSTM': BiLSTM,
            'Transformer': Transformer,
            'CNN+Transformer': CNN_Transformer,
            'CNN+Transformer_Parallel': CNN_Transformer_Parallel
        }
        
        # 训练每个模型并记录结果
        for model_name, ModelClass in models_to_compare.items():
            # 重新加载数据迭代器
            self.train_iter, self.dev_iter, self.test_iter = data.Iterator.splits(
                (self.train_iter.dataset, self.dev_iter.dataset, self.test_iter.dataset),
                batch_sizes=(self.args.batch_size, len(self.dev_iter.dataset), len(self.test_iter.dataset)),
                device=-1,
                sort_key=lambda x: len(x.text),
                sort_within_batch=False,
                repeat=False
            )
            
            # 创建模型
            model = ModelClass(self.args)
            print(f"\n{model_name} Model:")
            print(model)
            
            # 统计参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # 训练模型
            result = self.train_model(model, model_name)
            
            # 保存结果
            self.results[model_name] = {
                'best_dev_accuracy': result.best_dev_accuracy,
                'best_test_accuracy': result.best_test_accuracy,
                'best_epoch': result.best_epoch,
                'train_time': result.train_time,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            # 保存模型
            if not os.path.isdir(self.args.save_dir):
                os.makedirs(self.args.save_dir)
            save_path = os.path.join(self.args.save_dir, f'{model_name}_best.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        # 输出对比结果
        self.print_comparison()
        
        # 保存结果到文件
        self.save_results()
    
    def print_comparison(self):
        """打印对比结果"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'Best Dev Acc':<15} {'Best Test Acc':<15} {'Epoch':<10} {'Time(s)':<12} {'Params':<15}")
        print("-"*80)
        
        for model_name, result in self.results.items():
            print(f"{model_name:<20} "
                  f"{result['best_dev_accuracy']:<15.4f} "
                  f"{result['best_test_accuracy']:<15.4f} "
                  f"{result['best_epoch']:<10} "
                  f"{result['train_time']:<12.2f} "
                  f"{result['trainable_params']:<15,}")
        
        print("="*80)
        
        # 找出最佳模型
        best_model = max(self.results.items(), key=lambda x: x[1]['best_test_accuracy'])
        print(f"\nBest Model: {best_model[0]}")
        print(f"  Test Accuracy: {best_model[1]['best_test_accuracy']:.4f}%")
        print(f"  Dev Accuracy: {best_model[1]['best_dev_accuracy']:.4f}%")
    
    def save_results(self):
        """保存结果到文件"""
        result_file = os.path.join(self.args.save_dir, 'comparison_results.json')
        with open(result_file, 'w') as f:
            json.dump({
                'config': {
                    'epochs': self.args.epochs,
                    'batch_size': self.args.batch_size,
                    'lr': self.args.lr,
                    'dropout': self.args.dropout,
                    'embed_dim': self.args.embed_dim,
                    'lstm_hidden_dim': self.args.lstm_hidden_dim,
                    'seed': seed_num
                },
                'results': self.results
            }, f, indent=2)
        print(f"\nResults saved to {result_file}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Model Comparison: BiLSTM vs Transformer vs CNN+Transformer')
    
    # 数据参数
    parser.add_argument('--config_file', type=str, default='./Config/config.cfg',
                        help='configuration file')
    parser.add_argument('--max_len', type=int, default=100,
                        help='maximum sequence length')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=300,
                        help='word embedding dimension')
    parser.add_argument('--lstm_hidden_dim', type=int, default=300,
                        help='LSTM hidden dimension')
    parser.add_argument('--lstm_num_layers', type=int, default=1,
                        help='number of LSTM layers')
    parser.add_argument('--kernel_num', type=int, default=100,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3, 4, 5],
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--dropout_embed', type=float, default=0.5,
                        help='embedding dropout rate')
    parser.add_argument('--wide_conv', action='store_true', default=True,
                        help='use wide convolution')
    parser.add_argument('--init_weight', action='store_true', default=True,
                        help='init weight')
    parser.add_argument('--init_weight_value', type=float, default=2.0,
                        help='init weight value')
    parser.add_argument('--clip_max_norm', type=float, default=10.0,
                        help='gradient clip max norm')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=5,
                        help='early stopping patience')
    parser.add_argument('--Adam', action='store_true', default=True,
                        help='use Adam optimizer')
    parser.add_argument('--SGD', action='store_true', default=False,
                        help='use SGD optimizer')
    parser.add_argument('--Adadelta', action='store_true', default=False,
                        help='use Adadelta optimizer')
    
    # 日志参数
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many steps to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=100,
                        help='how many steps to wait before testing')
    parser.add_argument('--save_dir', type=str, default='./comparison_results',
                        help='where to save the snapshot')
    
    # 设备参数
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use CUDA')
    parser.add_argument('--device', type=int, default=-1,
                        help='device to use for iterate data, -1 means cpu')
    parser.add_argument('--threads', type=int, default=1,
                        help='number of threads for data loader')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = configurable.Configurable(args.config_file)
    
    # 从配置文件更新参数
    args.word_Embedding = config.word_Embedding
    args.word_Embedding_Path = config.word_Embedding_Path
    args.datafile_path = config.datafile_path
    args.min_freq = config.min_freq
    
    return args, config


if __name__ == '__main__':
    args, config = parse_args()
    
    # 设置线程数
    torch.set_num_threads(args.threads)
    
    # 运行对比实验
    comparison = ModelComparison(args, config)
    comparison.run_comparison()
