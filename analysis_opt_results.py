import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd

                                        
                                        
def load_results(base_path):
    results = defaultdict(lambda: defaultdict(dict))
        
    for algo_folder in os.listdir(base_path):
        if algo_folder.startswith('results_'):
            algo_path = os.path.join(base_path, algo_folder)
            for exp_folder in os.listdir(algo_path):
                if exp_folder.startswith('results_'):
                    exp_path = os.path.join(algo_path, exp_folder)
                    for exp2_folder in os.listdir(exp_path):
                        res_path = os.path.join(exp_path, exp2_folder)
                        for file in os.listdir(res_path):
                            if file.endswith('.jsonl'):
                                with open(os.path.join(res_path, file), 'r') as f:
                                    filename = file
                                    content = json.load(f)
                                    # Extract decision variables from filename
                                    x = []
                                    for key in ['lr', 'weight_decay', 'momentum', 'dropout_rate']:
                                        pattern = rf'({key}_)([\d.e-]+)'
                                        match = re.search(pattern, filename)
                                        if match:
                                            value = float(match.group(2))
                                            if key in ['lr', 'weight_decay']:
                                                x.append(np.log10(value))
                                            else:
                                                x.append(value)
                                    
                                    results[algo_folder][f"{exp_folder.split('_')[-1]}_{filename.split('_')[0]}"] = {
                                        'x': x,
                                        'test_standard_acc': content['test_standard_acc'],
                                        'test_robust_acc': np.mean([v for k, v in content.items() if k.startswith('test_') and k != 'test_standard_acc']),
                                        'val_acc': content['val_acc']
                                    }
                        
    return results


def plot_results_alg(results):
    fig, ax = plt.subplots()

    for algo, algo_results in results.items():
        best_param = max(algo_results.items(), key=lambda x: x[1]['val_acc'])
        param, param_results = best_param
        test_standard_acc = param_results['test_standard_acc']
        test_robust_acc = param_results['test_robust_acc']
        ax.scatter(test_robust_acc, test_standard_acc, label=algo.split('_')[1])

    ax.set_ylabel('Standard Accuracy')
    ax.set_xlabel('Test Robust Accuracy')
    ax.set_title(f"Results for Optimization")
    ax.legend()
    plt.savefig(f"all.png")
    plt.close()
    

def perform_anova(results):
    data = []
    for algo, algo_results in results.items():
        for param, param_results in algo_results.items():
            lr, weight_decay, momentum, dropout_rate = param_results['x']
            data.append({
                'lr': lr,
                'weight_decay': weight_decay,
                'momentum': momentum,
                'dropout_rate': dropout_rate,
                'test_standard_acc': param_results['test_standard_acc'],
                'test_robust_acc': param_results['test_robust_acc']
            })

    df = pd.DataFrame(data)
    
    # ANOVA for test_standard_acc
    model_standard = ols('test_standard_acc ~ lr + weight_decay + momentum + dropout_rate', data=df).fit()
    anova_standard = anova_lm(model_standard)
    
    # ANOVA for test_robust_acc
    model_robust = ols('test_robust_acc ~ lr + weight_decay + momentum + dropout_rate', data=df).fit()
    anova_robust = anova_lm(model_robust)
    
    print("ANOVA results for test_standard_acc:")
    print(anova_standard)
    print("\nANOVA results for test_robust_acc:")
    print(anova_robust)

def extract_params(param_string):
    params = {}
    for key in ['lr', 'weight_decay', 'momentum', 'dropout_rate']:
        pattern = rf'({key}_)([\d.e-]+)'
        match = re.search(pattern, param_string)
        if match:
            params[key] = float(match.group(2))
    
    return params.get('lr'), params.get('weight_decay'), params.get('momentum'), params.get('dropout_rate')

if __name__ == "__main__":
    base_path = './results'
    results = load_results(base_path)
    # plot_results_alg(results)
    perform_anova(results)
