import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def load_results(base_paths):
    all_results = {}
    for base_path in base_paths:
        results = defaultdict(lambda: defaultdict(dict))
        for file in os.listdir(base_path):
            if file.endswith('.jsonl'):
                with open(os.path.join(base_path, file), 'r') as f:
                    filename = file
                    content = json.load(f)
                    x = []
                    for key in ['lr', 'weight_decay', 'momentum', 'dropout_rate', 'batch_size']:
                        pattern = rf'({key}_)([\d.e-]+)'
                        match = re.search(pattern, filename)
                        if match:
                            value = float(match.group(2))
                            if key in ['lr', 'weight_decay']:
                                x.append(np.log10(value))
                            else:
                                x.append(value)
                    
                    results[base_path][f"{base_path.split('_')[-1]}_{filename.split('_')[0]}"] = {
                        'x': x,
                        'test_standard_acc': content['test_standard_acc'],
                        'test_robust_acc': np.mean([v for k, v in content.items() if k.startswith('test_') and k != 'test_standard_acc']),
                        'val_acc': content['val_acc']
                    }
        all_results[base_path] = results
    return all_results

def perform_anova(results):
    anova_results = {}
    for base_path, base_results in results.items():
        data = []
        for algo, algo_results in base_results.items():
            for param, param_results in algo_results.items():
                lr, weight_decay, momentum, dropout_rate, batch_size = param_results['x']
                data.append({
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'momentum': momentum,
                    'dropout_rate': dropout_rate,
                    'batch_size': batch_size,
                    'test_standard_acc': param_results['test_standard_acc'],
                    'test_robust_acc': param_results['test_robust_acc']
                })

        df = pd.DataFrame(data)
        
        # ANOVA for test_standard_acc
        model_standard = ols('test_standard_acc ~ lr + weight_decay + momentum + dropout_rate + batch_size', data=df).fit()
        anova_standard = anova_lm(model_standard)
        
        # ANOVA for test_robust_acc
        model_robust = ols('test_robust_acc ~ lr + weight_decay + momentum + dropout_rate + batch_size', data=df).fit()
        anova_robust = anova_lm(model_robust)
        
        anova_results[base_path] = {
            'standard': anova_standard,
            'robust': anova_robust
        }
    
    return anova_results

def plot_anova_results(anova_results):
    variables = ['lr', 'weight_decay', 'momentum', 'dropout_rate', 'batch_size']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    bar_width = 0.2
    index = np.arange(len(variables))
    
    for i, (base_path, results) in enumerate(anova_results.items()):
        standard_f_values = [results['standard'].loc[var, 'F'] for var in variables]
        robust_f_values = [results['robust'].loc[var, 'F'] for var in variables]
        
        label = base_path.split('/')[2]
        
        ax1.bar(index + i*bar_width, standard_f_values, bar_width, label=label)
        ax2.bar(index + i*bar_width, robust_f_values, bar_width, label=label)
    
    ax1.set_ylabel('F-value')
    ax1.set_title('Sensitivity Analysis for test_standard_acc')
    ax1.set_xticks(index + len(anova_results)*bar_width / 2)
    ax1.set_xticklabels(variables)
    ax1.legend()
    
    ax2.set_ylabel('F-value')
    ax2.set_title('Sensitivity Analysis for test_robust_acc')
    ax2.set_xticks(index + len(anova_results)*bar_width / 2)
    ax2.set_xticklabels(variables)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('anova_comparison.png')
    plt.close()

def plot_non_dominated_solutions(results):
    fig, ax = plt.subplots(figsize=(10, 8))

    for base_path, base_results in results.items():
        # 收集所有的 test_standard_acc 和 test_robust_acc
        f1 = []
        f2 = []
        for algo_results in base_results.values():
            for param_results in algo_results.values():
                f1.append(param_results['test_standard_acc'])
                f2.append(param_results['test_robust_acc'])
        
        # 执行非支配排序
        F = np.column_stack([-np.array(f1), -np.array(f2)])  # 注意：我们使用负值，因为我们要最大化这些指标
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        non_dominated = fronts[0]  # 获取第一个前沿（非支配解）

        # 绘制非支配解
        ax.scatter(np.array(f1)[non_dominated], np.array(f2)[non_dominated], label=base_path.split('/')[-1])

    ax.set_xlabel('Test Standard Accuracy')
    ax.set_ylabel('Test Robust Accuracy')
    ax.set_title('Non-dominated Solutions')
    ax.legend()
    plt.grid(True)
    plt.savefig('non_dominated_solutions.png')
    plt.close()

def get_best_files(base_path_list, metric):
    best_files = {}
    for base_path in base_path_list:
        best_value = float('-inf')
        best_file = None
        for file in os.listdir(base_path):
            if file.endswith('.jsonl'):
                file_path = os.path.join(base_path, file)
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    if metric in content:
                        value = content[metric]
                        if value > best_value:
                            best_value = value
                            best_file = file
        if best_file:
            best_files[base_path] = best_file
    return best_files

if __name__ == "__main__":
    import os

    results_dir = './results'
    base_paths = [os.path.join(results_dir, folder) for folder in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, folder))]
    
    base_paths = [
        './results/photometric_wideresnet28',     
        
        './results/cutout_wideresnet28',
        './results/geometric_wideresnet28',
        './results/mixup_wideresnet28',
        './results/withoutaug_wideresnet28',
        
        # './results/withoutaug_resnet50',
        # './results/withoutaug_resnet34',

        # './results/autoaugment_resnet18',
        # './results/mixup_resnet18',
        # './results/cutout_resnet18',
        # './results/geometric_resnet18',
        # './results/photometric_resnet18',
        # './results/withoutaug_resnet18',
    ]
    
    
    results = load_results(base_paths)
    anova_results = perform_anova(results)
    plot_anova_results(anova_results)
    
    # 绘制非支配解
    plot_non_dominated_solutions(results)
    
    # 获取每个 base_path 下 test_standard_acc 最高的文件
    best_files = get_best_files(base_paths, 'test_standard_acc')
    print("Files with highest test_standard_acc for each base_path:")
    for base_path, file in best_files.items():
        print(f"{base_path}: {file}")
    
    print("ANOVA results have been plotted and saved as 'anova_comparison.png'")
    print("Non-dominated solutions have been plotted and saved as 'non_dominated_solutions.png'")
