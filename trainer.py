import os
import sys
import subprocess
import shutil
import configparser
import time
import re

class ModelComparator:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_template = os.path.join(self.base_dir, 'Config', 'config.cfg')
        self.results_dir = os.path.join(self.base_dir, 'web_results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def create_config(self, model_name, hyperparams, output_config):
        config = configparser.ConfigParser()
        config.read(self.config_template)
        
        for section in config.sections():
            for key in config[section]:
                print(f"{section}.{key}: {config[section][key]}")
        
        config.set('Model', 'CNN', 'False')
        config.set('Model', 'LSTM', 'False')
        config.set('Model', 'GRU', 'False')
        config.set('Model', 'BiLSTM', 'False')
        config.set('Model', 'BiGRU', 'False')
        config.set('Model', 'CNN_LSTM', 'False')
        config.set('Model', 'CLSTM', 'False')
        config.set('Model', 'CBiLSTM', 'False')
        config.set('Model', 'CGRU', 'False')
        config.set('Model', 'CNN_BiLSTM', 'False')
        config.set('Model', 'CNN_BiGRU', 'False')
        config.set('Model', 'DEEP_CNN', 'False')
        config.set('Model', 'CNN_MUI', 'False')
        config.set('Model', 'DEEP_CNN_MUI', 'False')
        config.set('Model', 'HighWay_CNN', 'False')
        config.set('Model', 'HighWay_BiLSTM_1', 'False')
        config.set('Model', 'BiLSTM_1', 'False')
        
        config.set('Model', model_name, 'True')
        
        if 'learning_rate' in hyperparams:
            config.set('Optimizer', 'learning_rate', str(hyperparams['learning_rate']))
        if 'batch_size' in hyperparams:
            config.set('Train', 'batch_size', str(hyperparams['batch_size']))
        if 'epochs' in hyperparams:
            config.set('Train', 'epochs', str(hyperparams['epochs']))
        if 'embed_dim' in hyperparams:
            config.set('Model', 'embed_dim', str(hyperparams['embed_dim']))
        if 'kernel_num' in hyperparams:
            config.set('Model', 'kernel_num', str(hyperparams['kernel_num']))
        if 'dropout' in hyperparams:
            config.set('Model', 'dropout', str(hyperparams['dropout']))
        if 'lstm_hidden_dim' in hyperparams:
            config.set('Model', 'lstm_hidden_dim', str(hyperparams['lstm_hidden_dim']))
        if 'lstm_num_layers' in hyperparams:
            config.set('Model', 'lstm_num_layers', str(hyperparams['lstm_num_layers']))
        
        config.set('Train', 'cuda', 'False')
        
        with open(output_config, 'w') as f:
            config.write(f)
        print(f"Config created: {output_config}")
    
    def run_model(self, model_name, hyperparams):
        experiment_dir = os.path.join(self.results_dir, f"{model_name}_{int(time.time())}")
        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)
        
        config_path = os.path.join(experiment_dir, 'config.cfg')
        self.create_config(model_name, hyperparams, config_path)
        
        log_path = os.path.join(experiment_dir, 'training.log')
        
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                [sys.executable, 'main.py', '--config', config_path],
                cwd=self.base_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            process.wait()
        
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        return self.parse_results(log_content)
    
    def parse_results(self, log_content):
        results = {
            'best_accuracy': 0,
            'best_loss': float('inf'),
            'train_times': [],
            'epoch_data': [],
            'final_accuracy': 0
        }
        
        lines = log_content.split('\n')
        current_epoch = None
        epoch_train_time = None
        epoch_accuracy = None
        epoch_loss = None
        
        for line in lines:
            if 'Evaluation' in line and 'acc:' in line:
                match = re.search(r'acc:\s+([\d.]+)', line)
                if match:
                    epoch_accuracy = float(match.group(1))
                    results['final_accuracy'] = max(results['final_accuracy'], epoch_accuracy)
                    if epoch_accuracy > results['best_accuracy']:
                        results['best_accuracy'] = epoch_accuracy
            
            if 'Loss' in line and 'Time' in line:
                loss_match = re.search(r'Loss:\s+([\d.]+)', line)
                time_match = re.search(r'Time:\s+([\d.]+)', line)
                if loss_match:
                    epoch_loss = float(loss_match.group(1))
                    if epoch_loss < results['best_loss']:
                        results['best_loss'] = epoch_loss
                if time_match:
                    epoch_train_time = float(time_match.group(1))
                    results['train_times'].append(epoch_train_time)
            
            if 'Epoch' in line and 'Iter' in line:
                epoch_match = re.search(r'Epoch:\s*\[(\d+)/(\d+)\]', line)
                if epoch_match:
                    if current_epoch is not None and epoch_accuracy is not None:
                        results['epoch_data'].append({
                            'epoch': current_epoch,
                            'accuracy': epoch_accuracy,
                            'loss': epoch_loss if epoch_loss else 0,
                            'time': epoch_train_time if epoch_train_time else 0
                        })
                    current_epoch = int(epoch_match.group(1))
                    epoch_accuracy = None
                    epoch_loss = None
                    epoch_train_time = None
        
        if current_epoch is not None and epoch_accuracy is not None:
            results['epoch_data'].append({
                'epoch': current_epoch,
                'accuracy': epoch_accuracy if epoch_accuracy else results['best_accuracy'],
                'loss': epoch_loss if epoch_loss else results['best_loss'],
                'time': epoch_train_time if epoch_train_time else 0
            })
        
        print(f"Parsed results: {results}")
        return results
    
    def compare_models(self, model1, model2, hyperparams):
        print(f"Starting comparison between {model1} and {model2}")
        
        result1 = self.run_model(model1, hyperparams)
        result2 = self.run_model(model2, hyperparams)
        
        return {
            'model1': {'name': model1, 'data': result1},
            'model2': {'name': model2, 'data': result2}
        }


class MockModelComparator:
    def __init__(self):
        pass
    
    def compare_models(self, model1, model2, hyperparams):
        import time
        import random
        
        print(f"Mock training: {model1} vs {model2} with {hyperparams}")
        
        time.sleep(10)
        
        epochs = hyperparams.get('epochs', 10)
        
        def generate_mock_data(model_name, base_acc):
            data = []
            for i in range(1, epochs + 1):
                data.append({
                    'epoch': i,
                    'accuracy': min(base_acc + random.uniform(0.01, 0.03) * i, 0.98),
                    'loss': max(1.0 - random.uniform(0.05, 0.1) * i, 0.05),
                    'time': random.uniform(2.5, 4.5)
                })
            return {
                'best_accuracy': max([d['accuracy'] for d in data]),
                'best_loss': min([d['loss'] for d in data]),
                'epoch_data': data,
                'total_time': sum([d['time'] for d in data])
            }
        
        return {
            'model1': {'name': model1, 'data': generate_mock_data(model1, 0.65)},
            'model2': {'name': model2, 'data': generate_mock_data(model2, 0.7)}
        }
