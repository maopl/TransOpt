import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def analyze_exp_results(exp_dir="data/EXP_3"):
    # 存储不同方法的结果
    method_results = defaultdict(list)
    
    # 遍历EXP_3下的所有子文件夹
    for folder in os.listdir(exp_dir):
        folder_path = os.path.join(exp_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # 从文件夹名称中提取方法名（第三个下划线之后的部分）
        method = folder.split('_')[2]
        
        # 遍历文件夹中的所有jsonl文件
        for file in os.listdir(folder_path):
            if not file.endswith('.jsonl'):
                continue
                
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # 收集所有以test_开头的OOD指标（排除test_standard_acc）
                ood_metrics = []
                for key, value in data.items():
                    if key.startswith('test_') and key != 'test_standard_acc' and key != 'test_cifar10.1_acc' and key != 'test_cifar10.2_acc':
                        ood_metrics.append(value)
                
                # 计算OOD指标的平均值
                ood_avg = np.mean(ood_metrics)
                
                # 存储结果
                result = {
                    'standard_acc': data['test_standard_acc'],
                    'ood_avg': ood_avg,
                    'train_acc': data['train_acc'],
                    'val_acc': data['val_acc']
                }
                method_results[method].append(result)
    
    # 打印每种方法的平均结果
    print("Method Analysis Results:")
    print("-" * 80)
    print(f"{'Method':<15} {'Standard Acc':<15} {'OOD Avg':<15} {'Train Acc':<15} {'Val Acc':<15}")
    print("-" * 80)
    
    for method, results in method_results.items():
        avg_standard = np.mean([r['standard_acc'] for r in results])
        avg_ood = np.mean([r['ood_avg'] for r in results])
        avg_train = np.mean([r['train_acc'] for r in results])
        avg_val = np.mean([r['val_acc'] for r in results])
        
        print(f"{method:<15} {avg_standard:<15.4f} {avg_ood:<15.4f} {avg_train:<15.4f} {avg_val:<15.4f}")

def analyze_best_ood_results(exp_dir="data/EXP_3"):
    # 存储每种方法的最佳结果
    best_results = {}
    

    # 模拟的diversity指标（实际使用时替换为真实数据）
    diversity_scores = {
        "ERM": 11.06,
        "cutout": 8.86,
        "mixup": 11.07,
        "augment": 11.44,
        "non": 11.06
    }
    
    diversity_scores = {
        "ERM": 3.45,
        "cutout": 3.29,
        "mixup": 3.45,
        "augment": 3.95,
        "non": 3.45
    }
    
    # 遍历文件夹获取最佳结果
    for folder in os.listdir(exp_dir):
        folder_path = os.path.join(exp_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        method = folder.split('_')[2]
        best_ood_avg = -float('inf')
        best_result = None
        
        for file in os.listdir(folder_path):
            if not file.endswith('.jsonl'):
                continue
                
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # 计算OOD平均值
                ood_metrics = []
                for key, value in data.items():
                    if key.startswith('test_') and key != 'test_standard_acc' and key != 'test_cifar10.1_acc' and key != 'test_cifar10.2_acc':
                        ood_metrics.append(value)
                
                ood_avg = np.mean(ood_metrics)
                
                # 更新最佳结果
                if ood_avg > best_ood_avg:
                    best_ood_avg = ood_avg
                    best_result = {
                        'standard_acc': data['test_standard_acc'],
                        'ood_avg': ood_avg,
                        'train_acc': data['train_acc'],
                        'val_acc': data['val_acc']
                    }
        
        if best_result is not None:
            best_results[method] = best_result
    
    # 绘制折线图
    plt.figure(figsize=(12, 6))
    
    methods = list(best_results.keys())
    standard_accs = [best_results[m]['standard_acc'] for m in methods]
    ood_avgs = [best_results[m]['ood_avg'] for m in methods]
    diversity_values = [diversity_scores.get(m, 0.5) for m in methods]  # 默认值0.5
    
    plt.plot(methods, standard_accs, 'b-o', label='Standard Accuracy')
    plt.plot(methods, ood_avgs, 'r-o', label='OOD Average')
    plt.plot(methods, diversity_values, 'g-o', label='Diversity Score')
    
    plt.title('Best Results Comparison Across Methods')
    plt.xlabel('Methods')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('method_comparison.png')
    plt.close()
    
    # 打印最佳结果
    print("\nBest Results for Each Method:")
    print("-" * 80)
    print(f"{'Method':<15} {'Standard Acc':<15} {'OOD Avg':<15} {'Train Acc':<15} {'Val Acc':<15}")
    print("-" * 80)
    
    for method, result in best_results.items():
        print(f"{method:<15} {result['standard_acc']:<15.4f} {result['ood_avg']:<15.4f} "
              f"{result['train_acc']:<15.4f} {result['val_acc']:<15.4f}")
        
def plot_pareto_fronts_pymoo(base_dir = "data/EXP_3"):
    """
    使用pymoo库的非支配排序来分析和可视化EXP2的结果，并将数据保存为txt文件
    """
    def analyze_exp_data(exp_dir):
        """分析单个实验目录的数据，返回按方法分组的点"""
        folder_points = {}
        
        for folder in os.listdir(exp_dir):
            folder_path = os.path.join(exp_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            # 从文件夹名称中提取方法名
            method = folder.split('_')[2]
            
            if method not in folder_points:
                folder_points[method] = []
            
            # 遍历文件夹中的所有jsonl文件
            for file in os.listdir(folder_path):
                if not file.endswith('.jsonl'):
                    continue
                    
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # 计算OOD平均值
                    ood_metrics = []
                    for key, value in data.items():
                        if key == 'test_cifar10.1_acc' or key == 'test_cifar10.2_acc':
                            ood_metrics.append(value)
                    
                    ood_avg = np.mean(ood_metrics)
                    standard_acc = data['test_standard_acc']
                    
                    folder_points[method].append([standard_acc, ood_avg])
        
        # 将列表转换为numpy数组
        for method in folder_points:
            folder_points[method] = np.array(folder_points[method])
            
        return folder_points

    # 只分析EXP_2的数据
    exp2_method_points = analyze_exp_data(base_dir)

    # 创建图
    plt.figure(figsize=(10, 8))
    
    # 获取所有方法名称
    all_methods = sorted(list(exp2_method_points.keys()))
    # 为所有方法创建固定的颜色映射
    color_map = dict(zip(all_methods, plt.cm.rainbow(np.linspace(0, 1, len(all_methods)))))
    
    # 为每个方法保存数据点到txt文件
    for method, points in exp2_method_points.items():
        # 对每个方法的数据进行非支配排序
        nds = NonDominatedSorting()
        fronts = nds.do(-points)  # 取负是因为我们要最大化这些指标
        pareto_front = points[fronts[0]]
        non_pareto = points[~np.isin(np.arange(len(points)), fronts[0])]
        
        # # 保存非Pareto最优点到txt文件
        # np.savetxt(f'non_pareto_{method}.txt', non_pareto, 
        #           fmt='%.4f', delimiter=' ',
        #           header='% Standard_Accuracy OOD_Average',
        #           comments='')
        
        # 按x轴(Standard Accuracy)排序
        sort_idx = np.argsort(pareto_front[:, 0])
        pareto_front = pareto_front[sort_idx]
        
        # 保存排序后的Pareto最优点到txt文件
        np.savetxt(f'pareto_{method}.txt', pareto_front,
                  fmt='%.4f', delimiter=' ',
                  header='% Standard_Accuracy OOD_Average',
                  comments='')
        
        # 使用固定的颜色映射
        color = color_map[method]
        
        # 绘制该方法的所有点和帕累托前沿
        # plt.scatter(non_pareto[:, 0], non_pareto[:, 1], 
        #            c=[color], alpha=0.3, marker='.')
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], 
                   c=[color], s=100, label=method)
        plt.plot(pareto_front[:, 0], pareto_front[:, 1], 
                c=color, linestyle='--')
    
    plt.title('EXP2 Network Architecture Pareto Fronts')
    plt.xlabel('Standard Accuracy')
    plt.ylabel('OOD Average')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pareto_fronts_exp2.png')
    plt.close()
    
    # 打印统计信息
    print("\nEXP2 Statistics:")
    for method, points in exp2_method_points.items():
        nds = NonDominatedSorting()
        fronts = nds.do(-points)
        print(f"Method {method}:")
        print(f"  Total solutions: {len(points)}")
        print(f"  Pareto optimal solutions: {len(fronts[0])}")

def plot_individual_pareto_fronts(exp_dir="data/EXP_2"):
    """
    """
    # 遍历EXP_3下的所有子文件夹
    for folder in os.listdir(exp_dir):
        folder_path = os.path.join(exp_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 从文件夹名称中提取方法名
        method = folder.split('_')[2]
        points = []
        
        # 收集该文件夹中所有的数据点
        for file in os.listdir(folder_path):
            if not file.endswith('.jsonl'):
                continue
                
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # 计算OOD平均值
                ood_metrics = []
                for key, value in data.items():
                    if key.startswith('test_') and key != 'test_standard_acc' and key != 'test_cifar10.1_acc' and key != 'test_cifar10.2_acc':
                        ood_metrics.append(value)
                
                ood_avg = np.mean(ood_metrics)
                standard_acc = data['test_standard_acc']
                points.append([standard_acc, ood_avg])
        
        # 转换为numpy数组
        points = np.array(points)
        
        # 创建新图
        plt.figure(figsize=(10, 8))
        
        # 绘制所有数据点
        plt.scatter(points[:, 0], points[:, 1], c='blue', label='All points', alpha=0.6)
        
        # 计算pareto front
        nds = NonDominatedSorting()
        fronts = nds.do(-points)  # 取负是因为我们要最大化这些指标
        pareto_front = points[fronts[0]]
        non_pareto = points[~np.isin(np.arange(len(points)), fronts[0])]
        
        # 按x轴排序以便正确连线
        sort_idx = np.argsort(pareto_front[:, 0])
        pareto_front = pareto_front[sort_idx]
        
        # 绘制pareto front
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], 
                   c='red', s=100, label='Pareto front')
        plt.plot(pareto_front[:, 0], pareto_front[:, 1], 
                c='red', linestyle='--')
        
        plt.title(f'Pareto Analysis for {method}')
        plt.xlabel('Standard Accuracy')
        plt.ylabel('OOD Average')
        plt.grid(True)
        plt.legend()
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(f'pareto_front_{method}.png')
        plt.close()
        
        # 保存pareto front数据点到txt文件
        with open(f'pareto_{method}.txt', 'w') as f:
            f.write('% Standard_Accuracy OOD_Average\n')
            for point in pareto_front:
                f.write(f'{point[0]:.4f} {point[1]:.4f}\n')
                
        # 保存非pareto front数据点到txt文件
        with open(f'non_pareto_{method}.txt', 'w') as f:
            f.write('% Standard_Accuracy OOD_Average\n')
            for point in non_pareto:
                f.write(f'{point[0]:.4f} {point[1]:.4f}\n')
        
        # 打印统计信息
        print(f"\nStatistics for {method}:")
        print(f"Total solutions: {len(points)}")
        print(f"Pareto optimal solutions: {len(fronts[0])}")

if __name__ == "__main__":
    # analyze_exp_results()
    # analyze_best_ood_results()
    plot_pareto_fronts_pymoo()
    # plot_individual_pareto_fronts()
