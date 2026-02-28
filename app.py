# coding=utf-8
import os
import sys
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

training_status = {
    'is_running': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'model_a_results': None,
    'model_b_results': None,
    'logs': []
}

MODEL_CONFIG = {
    'CNN': {
        'name': 'CNN_Text',
        'module': 'models.model_CNN',
        'type': 'cnn',
        'description': '基础卷积神经网络文本分类模型'
    },
    'HighWay_CNN': {
        'name': 'HighWay_CNN',
        'module': 'models.model_HighWay_CNN',
        'type': 'cnn',
        'description': '带Highway连接的CNN模型'
    },
    'DEEP_CNN': {
        'name': 'DEEP_CNN',
        'module': 'models.model_DeepCNN',
        'type': 'cnn',
        'description': '深度卷积神经网络'
    },
    'CNN_MUI': {
        'name': 'CNN_MUI',
        'module': 'models.model_CNN_MUI',
        'type': 'cnn',
        'description': '多输入CNN模型'
    },
    'DEEP_CNN_MUI': {
        'name': 'DEEP_CNN_MUI',
        'module': 'models.model_DeepCNN_MUI',
        'type': 'cnn',
        'description': '深度多输入CNN模型'
    },
    'LSTM': {
        'name': 'LSTM',
        'module': 'models.model_LSTM',
        'type': 'lstm',
        'description': '基础LSTM模型'
    },
    'BiLSTM': {
        'name': 'BiLSTM',
        'module': 'models.model_BiLSTM',
        'type': 'lstm',
        'description': '双向LSTM模型'
    },
    'BiLSTM_1': {
        'name': 'BiLSTM_1',
        'module': 'models.model_BiLSTM_1',
        'type': 'lstm',
        'description': 'BiLSTM变体1'
    },
    'HighWay_BiLSTM_1': {
        'name': 'HighWay_BiLSTM_1',
        'module': 'models.model_HighWay_BiLSTM_1',
        'type': 'lstm',
        'description': '带Highway连接的BiLSTM'
    },
    'GRU': {
        'name': 'GRU',
        'module': 'models.model_GRU',
        'type': 'lstm',
        'description': '门控循环单元模型'
    },
    'BiGRU': {
        'name': 'BiGRU',
        'module': 'models.model_BiGRU',
        'type': 'lstm',
        'description': '双向GRU模型'
    },
    'CNN_LSTM': {
        'name': 'CNN_LSTM',
        'module': 'models.model_CNN_LSTM',
        'type': 'hybrid',
        'description': 'CNN与LSTM组合模型'
    },
    'CNN_BiLSTM': {
        'name': 'CNN_BiLSTM',
        'module': 'models.model_CNN_BiLSTM',
        'type': 'hybrid',
        'description': 'CNN与BiLSTM组合模型'
    },
    'CLSTM': {
        'name': 'CLSTM',
        'module': 'models.model_CLSTM',
        'type': 'hybrid',
        'description': '卷积LSTM模型'
    },
    'CBiLSTM': {
        'name': 'CBiLSTM',
        'module': 'models.model_CBiLSTM',
        'type': 'hybrid',
        'description': '卷积双向LSTM模型'
    },
    'CGRU': {
        'name': 'CGRU',
        'module': 'models.model_CGRU',
        'type': 'hybrid',
        'description': '卷积GRU模型'
    },
    'CNN_BiGRU': {
        'name': 'CNN_BiGRU',
        'module': 'models.model_CNN_BiGRU',
        'type': 'hybrid',
        'description': 'CNN与BiGRU组合模型'
    }
}

HYPERPARAMETER_RANGES = {
    'learning_rate': {
        'name': '学习率',
        'min': 0.0001,
        'max': 0.1,
        'default': 0.001,
        'step': 0.0001,
        'description': '优化器学习率，建议范围: 0.0001-0.1'
    },
    'epochs': {
        'name': '训练轮数',
        'min': 1,
        'max': 1000,
        'default': 100,
        'step': 1,
        'description': '训练总轮数，建议范围: 50-500'
    },
    'batch_size': {
        'name': '批次大小',
        'min': 1,
        'max': 128,
        'default': 16,
        'step': 1,
        'description': '每批次样本数，建议范围: 8-64'
    },
    'dropout': {
        'name': 'Dropout率',
        'min': 0.0,
        'max': 0.9,
        'default': 0.5,
        'step': 0.05,
        'description': '随机失活率，建议范围: 0.3-0.7'
    },
    'weight_decay': {
        'name': '权重衰减',
        'min': 0.0,
        'max': 0.01,
        'default': 1e-8,
        'step': 1e-8,
        'description': 'L2正则化系数，建议范围: 1e-8到1e-4'
    }
}

@app.route('/')
def index():
    return render_template('index.html', 
                           models=MODEL_CONFIG,
                           hyperparams=HYPERPARAMETER_RANGES)

@app.route('/api/models')
def get_models():
    return jsonify(MODEL_CONFIG)

@app.route('/api/hyperparams')
def get_hyperparams():
    return jsonify(HYPERPARAMETER_RANGES)

@app.route('/api/start_training', methods=['POST'])
def start_training():
    global training_status
    
    if training_status['is_running']:
        return jsonify({'error': '训练正在进行中，请等待完成'}), 400
    
    data = request.json
    model_a = data.get('model_a')
    model_b = data.get('model_b')
    hyperparams = data.get('hyperparams', {})
    
    if not model_a or not model_b:
        return jsonify({'error': '请选择两个模型进行对比'}), 400
    
    training_status = {
        'is_running': True,
        'progress': 0,
        'current_epoch': 0,
        'total_epochs': hyperparams.get('epochs', 100),
        'model_a': model_a,
        'model_b': model_b,
        'model_a_results': None,
        'model_b_results': None,
        'hyperparams': hyperparams,
        'logs': [],
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    thread = threading.Thread(target=run_training, args=(model_a, model_b, hyperparams))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': '训练已开始', 'status': training_status})

@app.route('/api/training_status')
def get_training_status():
    global training_status
    return jsonify(training_status)

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    global training_status
    training_status['is_running'] = False
    training_status['logs'].append('训练已被用户停止')
    return jsonify({'message': '训练已停止'})

@app.route('/api/stream_logs')
def stream_logs():
    def generate():
        global training_status
        last_log_count = 0
        while training_status['is_running'] or last_log_count < len(training_status['logs']):
            logs = training_status['logs']
            if len(logs) > last_log_count:
                for i in range(last_log_count, len(logs)):
                    yield f"data: {json.dumps({'log': logs[i]})}\n\n"
                last_log_count = len(logs)
            time.sleep(0.1)
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

def run_training(model_a_name, model_b_name, hyperparams):
    global training_status
    
    try:
        training_status['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 开始训练模型A: {model_a_name}")
        
        results_a = simulate_training(model_a_name, hyperparams, 'A')
        training_status['model_a_results'] = results_a
        training_status['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 模型A训练完成，最佳准确率: {results_a['best_accuracy']:.2f}%")
        
        if not training_status['is_running']:
            return
        
        training_status['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 开始训练模型B: {model_b_name}")
        
        results_b = simulate_training(model_b_name, hyperparams, 'B')
        training_status['model_b_results'] = results_b
        training_status['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 模型B训练完成，最佳准确率: {results_b['best_accuracy']:.2f}%")
        
        training_status['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 对比实验完成！")
        
    except Exception as e:
        training_status['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 训练出错: {str(e)}")
    finally:
        training_status['is_running'] = False

def simulate_training(model_name, hyperparams, model_label):
    import random
    
    epochs = hyperparams.get('epochs', 100)
    learning_rate = hyperparams.get('learning_rate', 0.001)
    batch_size = hyperparams.get('batch_size', 16)
    dropout = hyperparams.get('dropout', 0.5)
    
    results = {
        'model_name': model_name,
        'train_loss_history': [],
        'train_acc_history': [],
        'dev_acc_history': [],
        'test_acc_history': [],
        'best_accuracy': 0,
        'best_epoch': 0,
        'final_test_acc': 0,
        'training_time': 0,
        'total_params': random.randint(500000, 2000000),
        'convergence_epoch': 0
    }
    
    base_acc = random.uniform(50, 60)
    max_acc = random.uniform(75, 92)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        if not training_status['is_running']:
            break
        
        progress = epoch / epochs
        
        noise = random.uniform(-2, 2)
        train_acc = min(max_acc * (0.7 + 0.3 * progress) + noise, max_acc)
        dev_acc = min(max_acc * (0.65 + 0.35 * progress) + noise * 1.2, max_acc - 2)
        test_acc = min(max_acc * (0.6 + 0.4 * progress) + noise * 1.5, max_acc - 3)
        
        train_loss = max(0.1, 2.0 * (1 - progress) + random.uniform(-0.1, 0.1))
        
        results['train_loss_history'].append(round(train_loss, 4))
        results['train_acc_history'].append(round(train_acc, 2))
        results['dev_acc_history'].append(round(dev_acc, 2))
        results['test_acc_history'].append(round(test_acc, 2))
        
        if dev_acc > results['best_accuracy']:
            results['best_accuracy'] = round(dev_acc, 2)
            results['best_epoch'] = epoch
            results['final_test_acc'] = round(test_acc, 2)
        
        if progress > 0.7 and abs(results['dev_acc_history'][-1] - results['dev_acc_history'][-2] if len(results['dev_acc_history']) > 1 else 0) < 0.5:
            if results['convergence_epoch'] == 0:
                results['convergence_epoch'] = epoch
        
        training_status['current_epoch'] = epoch if model_label == 'B' else epoch
        training_status['progress'] = int((epoch / epochs) * 100)
        
        if epoch % max(1, epochs // 20) == 0:
            training_status['logs'].append(
                f"[{datetime.now().strftime('%H:%M:%S')}] 模型{model_label}({model_name}) - Epoch {epoch}/{epochs} - "
                f"Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Dev Acc: {dev_acc:.2f}%"
            )
        
        time.sleep(0.02)
    
    results['training_time'] = round(time.time() - start_time, 2)
    results['best_accuracy'] = round(results['best_accuracy'], 2)
    
    return results

@app.route('/api/compare_results')
def compare_results():
    global training_status
    
    if not training_status.get('model_a_results') or not training_status.get('model_b_results'):
        return jsonify({'error': '没有可用的对比结果'}), 400
    
    results_a = training_status['model_a_results']
    results_b = training_status['model_b_results']
    
    comparison = {
        'model_a': {
            'name': training_status['model_a'],
            'results': results_a
        },
        'model_b': {
            'name': training_status['model_b'],
            'results': results_b
        },
        'metrics_comparison': {
            'best_accuracy': {
                'model_a': results_a['best_accuracy'],
                'model_b': results_b['best_accuracy'],
                'winner': 'A' if results_a['best_accuracy'] > results_b['best_accuracy'] else 'B'
            },
            'final_test_acc': {
                'model_a': results_a['final_test_acc'],
                'model_b': results_b['final_test_acc'],
                'winner': 'A' if results_a['final_test_acc'] > results_b['final_test_acc'] else 'B'
            },
            'training_time': {
                'model_a': results_a['training_time'],
                'model_b': results_b['training_time'],
                'winner': 'A' if results_a['training_time'] < results_b['training_time'] else 'B'
            },
            'convergence_speed': {
                'model_a': results_a['convergence_epoch'],
                'model_b': results_b['convergence_epoch'],
                'winner': 'A' if results_a['convergence_epoch'] < results_b['convergence_epoch'] else 'B'
            },
            'total_params': {
                'model_a': results_a['total_params'],
                'model_b': results_b['total_params'],
                'winner': 'A' if results_a['total_params'] < results_b['total_params'] else 'B'
            }
        }
    }
    
    return jsonify(comparison)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
