from flask import Flask, render_template, request, jsonify
import os
import threading
import json
from datetime import datetime

app = Flask(__name__)

AVAILABLE_MODELS = [
    'CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU',
    'CNN_LSTM', 'CLSTM', 'CBiLSTM', 'CGRU',
    'CNN_BiLSTM', 'CNN_BiGRU', 'DEEP_CNN', 'CNN_MUI'
]

HYPERPARAMS = {
    'learning_rate': {'default': 0.001, 'min': 0.0001, 'max': 0.1, 'step': 0.0001, 'type': 'float', 'desc': '学习率'},
    'batch_size': {'default': 32, 'min': 8, 'max': 256, 'step': 8, 'type': 'int', 'desc': '批次大小'},
    'epochs': {'default': 10, 'min': 1, 'max': 100, 'step': 1, 'type': 'int', 'desc': '训练轮数'},
    'embed_dim': {'default': 100, 'min': 50, 'max': 500, 'step': 50, 'type': 'int', 'desc': '词向量维度'},
    'kernel_num': {'default': 100, 'min': 10, 'max': 500, 'step': 10, 'type': 'int', 'desc': '卷积核数量'},
    'dropout': {'default': 0.5, 'min': 0.1, 'max': 0.9, 'step': 0.1, 'type': 'float', 'desc': 'Dropout率'},
    'weight_decay': {'default': 0.0, 'min': 0.0, 'max': 0.01, 'step': 0.0001, 'type': 'float', 'desc': '权重衰减(L2正则化)'},
    'clip_max_norm': {'default': 3, 'min': 1, 'max': 10, 'step': 1, 'type': 'int', 'desc': '梯度裁剪阈值'}
}

training_results = {}
training_status = {}

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS, hyperparams=HYPERPARAMS)

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.json
    model1 = data.get('model1')
    model2 = data.get('model2')
    hyperparams = data.get('hyperparams', {})
    
    if model1 == model2:
        return jsonify({'error': '请选择两个不同的模型进行对比'}), 400
    
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_status[experiment_id] = 'running'
    
    thread = threading.Thread(target=run_training, args=(experiment_id, model1, model2, hyperparams))
    thread.daemon = True
    thread.start()
    
    return jsonify({'experiment_id': experiment_id})

@app.route('/api/status/<experiment_id>')
def status(experiment_id):
    result = training_results.get(experiment_id, {})
    current_status = training_status.get(experiment_id, 'unknown')
    return jsonify({
        'status': current_status,
        'result': result
    })

def run_training(experiment_id, model1, model2, hyperparams):
    try:
        from trainer import MockModelComparator
        comparator = MockModelComparator()
        result = comparator.compare_models(model1, model2, hyperparams)
        training_results[experiment_id] = result
        training_status[experiment_id] = 'completed'
    except Exception as e:
        training_status[experiment_id] = 'failed'
        training_results[experiment_id] = {'error': str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)