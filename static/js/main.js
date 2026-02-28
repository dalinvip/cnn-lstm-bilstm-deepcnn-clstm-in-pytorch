let lossChart, accChart, devAccChart, radarChart;
let trainingInterval;

const modelDescriptions = {
    'CNN': '基础卷积神经网络，使用多尺寸卷积核进行文本特征提取',
    'HighWay_CNN': '在CNN基础上添加Highway连接，改善梯度流动',
    'DEEP_CNN': '深度卷积网络，多层卷积堆叠提取深层特征',
    'CNN_MUI': '多输入CNN，支持多种嵌入方式的融合',
    'DEEP_CNN_MUI': '深度多输入CNN，结合深度网络与多输入优势',
    'LSTM': '标准长短期记忆网络，适合序列建模',
    'BiLSTM': '双向LSTM，同时捕获前后文信息',
    'BiLSTM_1': 'BiLSTM变体，改进的架构设计',
    'HighWay_BiLSTM_1': '带Highway连接的BiLSTM，优化深层网络训练',
    'GRU': '门控循环单元，参数量少于LSTM',
    'BiGRU': '双向GRU，高效的双向序列建模',
    'CNN_LSTM': 'CNN+LSTM组合，局部特征与序列建模结合',
    'CNN_BiLSTM': 'CNN+BiLSTM组合，双向序列建模能力',
    'CLSTM': '卷积LSTM，将卷积操作融入LSTM单元',
    'CBiLSTM': '卷积双向LSTM，结合卷积与双向LSTM优势',
    'CGRU': '卷积GRU，轻量级的混合架构',
    'CNN_BiGRU': 'CNN+BiGRU组合，高效的双向混合模型'
};

document.getElementById('model-a-select').addEventListener('change', function() {
    const model = this.value;
    const infoDiv = document.getElementById('model-a-info');
    if (model && modelDescriptions[model]) {
        infoDiv.innerHTML = '<strong>模型说明:</strong> ' + modelDescriptions[model];
    } else {
        infoDiv.innerHTML = '';
    }
});

document.getElementById('model-b-select').addEventListener('change', function() {
    const model = this.value;
    const infoDiv = document.getElementById('model-b-info');
    if (model && modelDescriptions[model]) {
        infoDiv.innerHTML = '<strong>模型说明:</strong> ' + modelDescriptions[model];
    } else {
        infoDiv.innerHTML = '';
    }
});

function getHyperparameters() {
    return {
        learning_rate: parseFloat(document.getElementById('learning_rate').value),
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
        dropout: parseFloat(document.getElementById('dropout').value),
        weight_decay: parseFloat(document.getElementById('weight_decay').value)
    };
}

function startTraining() {
    const modelA = document.getElementById('model-a-select').value;
    const modelB = document.getElementById('model-b-select').value;
    
    if (!modelA || !modelB) {
        alert('请选择两个模型进行对比！');
        return;
    }
    
    if (modelA === modelB) {
        alert('请选择不同的模型进行对比！');
        return;
    }
    
    const hyperparams = getHyperparameters();
    
    document.getElementById('start-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
    document.getElementById('progress-section').style.display = 'block';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('logs-output').innerHTML = '';
    
    fetch('/api/start_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_a: modelA,
            model_b: modelB,
            hyperparams: hyperparams
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
        } else {
            startStatusPolling();
            startLogStreaming();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('启动训练失败');
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
    });
}

function stopTraining() {
    fetch('/api/stop_training', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        stopStatusPolling();
    });
}

function startStatusPolling() {
    trainingInterval = setInterval(() => {
        fetch('/api/training_status')
            .then(response => response.json())
            .then(data => {
                updateProgress(data);
                
                if (!data.is_running && data.model_a_results && data.model_b_results) {
                    stopStatusPolling();
                    showResults(data);
                }
            });
    }, 500);
}

function stopStatusPolling() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
}

function updateProgress(data) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const currentEpoch = document.getElementById('current-epoch');
    const trainingStatus = document.getElementById('training-status');
    
    progressFill.style.width = data.progress + '%';
    progressText.textContent = data.progress + '%';
    currentEpoch.textContent = '当前轮次: ' + data.current_epoch + '/' + data.total_epochs;
    trainingStatus.textContent = '状态: ' + (data.is_running ? '训练中...' : '已完成');
}

function startLogStreaming() {
    const logsOutput = document.getElementById('logs-output');
    const eventSource = new EventSource('/api/stream_logs');
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.done) {
            eventSource.close();
            return;
        }
        
        if (data.log) {
            const logLine = document.createElement('div');
            logLine.textContent = data.log;
            logsOutput.appendChild(logLine);
            logsOutput.scrollTop = logsOutput.scrollHeight;
        }
    };
    
    eventSource.onerror = function() {
        eventSource.close();
    };
}

function showResults(data) {
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    document.getElementById('results-section').style.display = 'block';
    
    fetch('/api/compare_results')
        .then(response => response.json())
        .then(comparison => {
            displayMetrics(comparison);
            displayCharts(comparison);
            displayTable(comparison);
        });
}

function displayMetrics(comparison) {
    const metrics = comparison.metrics_comparison;
    
    document.querySelector('#metric-acc-a .metric-value').textContent = 
        metrics.best_accuracy.model_a.toFixed(2) + '%';
    document.querySelector('#metric-acc-b .metric-value').textContent = 
        metrics.best_accuracy.model_b.toFixed(2) + '%';
    
    document.querySelector('#metric-test-a .metric-value').textContent = 
        metrics.final_test_acc.model_a.toFixed(2) + '%';
    document.querySelector('#metric-test-b .metric-value').textContent = 
        metrics.final_test_acc.model_b.toFixed(2) + '%';
    
    document.querySelector('#metric-time-a .metric-value').textContent = 
        metrics.training_time.model_a.toFixed(2) + 's';
    document.querySelector('#metric-time-b .metric-value').textContent = 
        metrics.training_time.model_b.toFixed(2) + 's';
    
    document.querySelector('#metric-conv-a .metric-value').textContent = 
        metrics.convergence_speed.model_a || '--';
    document.querySelector('#metric-conv-b .metric-value').textContent = 
        metrics.convergence_speed.model_b || '--';
    
    document.querySelector('#metric-params-a .metric-value').textContent = 
        formatNumber(metrics.total_params.model_a);
    document.querySelector('#metric-params-b .metric-value').textContent = 
        formatNumber(metrics.total_params.model_b);
    
    highlightWinners(metrics);
}

function highlightWinners(metrics) {
    document.querySelectorAll('.metric-a, .metric-b').forEach(el => {
        el.classList.remove('winner-a', 'winner-b');
    });
    
    if (metrics.best_accuracy.winner === 'A') {
        document.querySelector('#metric-acc-a').classList.add('winner-a');
    } else {
        document.querySelector('#metric-acc-b').classList.add('winner-b');
    }
    
    if (metrics.final_test_acc.winner === 'A') {
        document.querySelector('#metric-test-a').classList.add('winner-a');
    } else {
        document.querySelector('#metric-test-b').classList.add('winner-b');
    }
    
    if (metrics.training_time.winner === 'A') {
        document.querySelector('#metric-time-a').classList.add('winner-a');
    } else {
        document.querySelector('#metric-time-b').classList.add('winner-b');
    }
}

function displayCharts(comparison) {
    const resultsA = comparison.model_a.results;
    const resultsB = comparison.model_b.results;
    const labels = Array.from({length: resultsA.train_loss_history.length}, (_, i) => i + 1);
    
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                labels: {
                    color: '#fff'
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#888' },
                grid: { color: 'rgba(255,255,255,0.1)' }
            },
            y: {
                ticks: { color: '#888' },
                grid: { color: 'rgba(255,255,255,0.1)' }
            }
        }
    };
    
    if (lossChart) lossChart.destroy();
    lossChart = new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: comparison.model_a.name + ' - Train Loss',
                data: resultsA.train_loss_history,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: comparison.model_b.name + ' - Train Loss',
                data: resultsB.train_loss_history,
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });
    
    if (accChart) accChart.destroy();
    accChart = new Chart(document.getElementById('accChart'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: comparison.model_a.name + ' - Train Acc',
                data: resultsA.train_acc_history,
                borderColor: '#00d4ff',
                tension: 0.4
            }, {
                label: comparison.model_b.name + ' - Train Acc',
                data: resultsB.train_acc_history,
                borderColor: '#00ff88',
                tension: 0.4
            }]
        },
        options: chartOptions
    });
    
    if (devAccChart) devAccChart.destroy();
    devAccChart = new Chart(document.getElementById('devAccChart'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: comparison.model_a.name + ' - Dev Acc',
                data: resultsA.dev_acc_history,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: comparison.model_b.name + ' - Dev Acc',
                data: resultsB.dev_acc_history,
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });
    
    if (radarChart) radarChart.destroy();
    const metrics = comparison.metrics_comparison;
    radarChart = new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
            labels: ['最佳准确率', '测试准确率', '训练速度', '收敛速度', '参数效率'],
            datasets: [{
                label: comparison.model_a.name,
                data: [
                    normalizeAcc(metrics.best_accuracy.model_a),
                    normalizeAcc(metrics.final_test_acc.model_a),
                    normalizeTime(metrics.training_time.model_a),
                    normalizeConvergence(metrics.convergence_speed.model_a),
                    normalizeParams(metrics.total_params.model_a)
                ],
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.2)'
            }, {
                label: comparison.model_b.name,
                data: [
                    normalizeAcc(metrics.best_accuracy.model_b),
                    normalizeAcc(metrics.final_test_acc.model_b),
                    normalizeTime(metrics.training_time.model_b),
                    normalizeConvergence(metrics.convergence_speed.model_b),
                    normalizeParams(metrics.total_params.model_b)
                ],
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.2)'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: { color: '#fff' }
                }
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.2)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#fff' },
                    ticks: { 
                        color: '#888',
                        backdropColor: 'transparent'
                    }
                }
            }
        }
    });
}

function displayTable(comparison) {
    const metrics = comparison.metrics_comparison;
    const tbody = document.getElementById('results-tbody');
    
    const rows = [
        {
            name: '最佳验证准确率',
            a: metrics.best_accuracy.model_a.toFixed(2) + '%',
            b: metrics.best_accuracy.model_b.toFixed(2) + '%',
            winner: metrics.best_accuracy.winner
        },
        {
            name: '最终测试准确率',
            a: metrics.final_test_acc.model_a.toFixed(2) + '%',
            b: metrics.final_test_acc.model_b.toFixed(2) + '%',
            winner: metrics.final_test_acc.winner
        },
        {
            name: '训练时间',
            a: metrics.training_time.model_a.toFixed(2) + 's',
            b: metrics.training_time.model_b.toFixed(2) + 's',
            winner: metrics.training_time.winner
        },
        {
            name: '收敛轮次',
            a: metrics.convergence_speed.model_a || '--',
            b: metrics.convergence_speed.model_b || '--',
            winner: metrics.convergence_speed.winner
        },
        {
            name: '模型参数量',
            a: formatNumber(metrics.total_params.model_a),
            b: formatNumber(metrics.total_params.model_b),
            winner: metrics.total_params.winner
        }
    ];
    
    tbody.innerHTML = rows.map(row => 
        '<tr>' +
            '<td>' + row.name + '</td>' +
            '<td class="' + (row.winner === 'A' ? 'winner-text-a' : '') + '">' + row.a + '</td>' +
            '<td class="' + (row.winner === 'B' ? 'winner-text-b' : '') + '">' + row.b + '</td>' +
            '<td class="' + (row.winner === 'A' ? 'winner-text-a' : 'winner-text-b') + '">' +
                (row.winner === 'A' ? 'Model A' : 'Model B') +
            '</td>' +
        '</tr>'
    ).join('');
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    }
    return num.toString();
}

function normalizeAcc(value) {
    return Math.min(100, Math.max(0, value));
}

function normalizeTime(value) {
    return Math.max(0, 100 - value);
}

function normalizeConvergence(value) {
    if (!value) return 50;
    return Math.max(0, 100 - value);
}

function normalizeParams(value) {
    return Math.max(0, 100 - (value / 20000));
}
