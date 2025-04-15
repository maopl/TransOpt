import os
import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_experiment_data(base_dir):
    results = {}

    # 遍历 compare 文件夹中的所有子文件夹
    for algo_dir in os.listdir(base_dir):
        algo_path = os.path.join(base_dir, algo_dir)
        if os.path.isdir(algo_path):
            results[algo_dir] = {}

            # 遍历每个算法子文件夹中的应用子文件夹
            for app_dir in os.listdir(algo_path):
                app_path = os.path.join(algo_path, app_dir)
                if os.path.isdir(app_path):
                    results[algo_dir][app_dir] = []

                    # 遍历每个应用子文件夹中的 JSONL 文件
                    for file_name in os.listdir(app_path):
                        if file_name.endswith('.jsonl'):
                            file_path = os.path.join(app_path, file_name)
                            with open(file_path, 'r') as file:
                                data = json.loads(file.read())  # Changed from file_path to file.read()
                                # 提取 test_standard_acc 和 test 开头非 standard 的数据
                                test_standard_acc = data.get('test_standard_acc')  # Fixed typo in key name
                                test_metrics = {k: v for k, v in data.items() if k.startswith('test') and 'standard' not in k}
                                results[algo_dir][app_dir].append({
                                    'test_standard_acc': test_standard_acc,  # Fixed key name
                                    'test_metrics': test_metrics
                                })

    # 绘制每个算法在不同应用上的表现
    for algo, apps in results.items():
        for app, metrics in apps.items():
            test_standard_acc_values = [m['test_standard_acc'] for m in metrics]
            # Calculate mean of all test metrics for each experiment
            test_robust_acc_values = [np.mean(list(m['test_metrics'].values())) for m in metrics]

            plt.figure(figsize=(10, 5))
            plt.plot([x + 0.5 for x in test_standard_acc_values], label='test_standard_acc', marker='o')
            plt.plot([x + 0.5 for x in test_robust_acc_values], label='test_robust_acc', marker='x')

            plt.title(f'Algorithm: {algo} - Application: {app}')
            plt.xlabel('Experiment Index')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{algo}_{app}.png')

def plot_specific_experiment_data(specific_dir):
    results = []

    # 遍历指定文件夹中的 JSONL 文件
    for file_name in os.listdir(specific_dir):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(specific_dir, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    # 提取 tes_standardacc 和 test 开头非 standard 的数据
                    tes_standardacc = data.get('tes_standardacc')
                    test_metrics = {k: v for k, v in data.items() if k.startswith('test') and 'standard' not in k}
                    results.append({
                        'tes_standardacc': tes_standardacc,
                        'test_metrics': test_metrics
                    })

    # 绘制数据
    tes_standardacc_values = [m['tes_standardacc'] for m in results]
    test_metric_values = [list(m['test_metrics'].values()) for m in results]

    plt.figure(figsize=(10, 5))
    plt.plot(tes_standardacc_values, label='tes_standardacc', marker='o')
    for i, test_values in enumerate(zip(*test_metric_values)):
        plt.plot(test_values, label=f'test_metric_{i}', marker='x')

    plt.title('Specific Experiment Data')
    plt.xlabel('Experiment Index')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用示例
base_directory = 'compare'
analyze_experiment_data(base_directory)

# 绘制特定路径下的数据
specific_directory = os.path.join('EXP_1', 'results_bohb', 'resnet', 'results_1', 'bohb_ERM_resnet_18_RobCifar10_42')
plot_specific_experiment_data(specific_directory)
