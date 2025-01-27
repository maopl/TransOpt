import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def process_exp_data(root_dir="./data/EXP_1"):
    # 用于存储最终结果的字典
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # 遍历第一级目录 (算法)
    for algo_dir in os.listdir(root_dir):
        algo_path = os.path.join(root_dir, algo_dir)
        if not os.path.isdir(algo_path):
            continue
            
        # 遍历第二级目录 (神经网络架构)
        for arch_dir in os.listdir(algo_path):
            arch_path = os.path.join(algo_path, arch_dir)
            if not os.path.isdir(arch_path):
                continue
                
            # 遍历第三级目录
            for exp_dir in os.listdir(arch_path):
                exp_path = os.path.join(arch_path, exp_dir)
                if not os.path.isdir(exp_path):
                    continue
                    
                # 解析数据集名称 (倒数第二个部分)
                parts = exp_dir.split('_')
                dataset = parts[-2]
                
                # 处理该目录下的所有jsonl文件
                in_dist_accs = []
                ood_accs = []
                
                for file in os.listdir(exp_path):
                    if not file.endswith('.jsonl'):
                        continue
                        
                    with open(os.path.join(exp_path, file), 'r') as f:
                        data = json.load(f)
                        
                        # 获取in-distribution准确率
                        in_dist_acc = data['test_standard_acc']
                        
                        # 计算OOD准确率 (所有以test_开头但不是standard的指标的平均值)
                        ood_metrics = [v for k, v in data.items() 
                                    if k.startswith('test_') and k != 'test_standard_acc']
                        ood_acc = np.mean(ood_metrics) if ood_metrics else None
                        
                        if ood_acc is not None:
                            in_dist_accs.append(in_dist_acc)
                            ood_accs.append(ood_acc)
                
                # 将结果存入字典
                if in_dist_accs and ood_accs:
                    algo_name = algo_dir.replace('results_', '')  # 移除'results_'前缀
                    results[algo_name][arch_dir][dataset] = {
                        'in_distribution_acc': in_dist_accs,
                        'ood_generalization': ood_accs
                    }
    
    return results
def plot_overfitting_curves(results, save_dir="./plots/overfitting"):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个数据集创建单独的图
    datasets = set()
    for algo_data in results.values():
        for arch_data in algo_data.values():
            datasets.update(arch_data.keys())
    
    for dataset in datasets:
        # 为每个算法-架构组合画一个单独的图
        for algo, algo_data in results.items():
            for arch, arch_data in algo_data.items():
                if dataset not in arch_data:
                    continue
                    
                plt.figure(figsize=(10, 6))
                
                metrics = arch_data[dataset]
                in_dist = metrics['in_distribution_acc']
                ood = metrics['ood_generalization']
                # 计算累计最大值
                max_in_dist = []
                corresponding_ood = []
                current_max = float('-inf')
                
                for i in range(len(in_dist)):
                    if in_dist[i] > current_max:
                        current_max = in_dist[i]
                        max_in_dist.append(current_max)
                        corresponding_ood.append(ood[i])
                
                # 绘制曲线
                label = f"{algo}-{arch}"
                plt.plot(max_in_dist, corresponding_ood, marker='o', label=label, markersize=4)
                
                plt.xlabel('In-distribution Accuracy')
                plt.ylabel('OOD Generalization')
                plt.title(f'Overfitting Analysis - {dataset} - {label}')
                plt.grid(True)
                plt.tight_layout()
                
                # 保存图片
                save_path = os.path.join(save_dir, f'overfitting_{dataset}_{algo}_{arch}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()

def plot_overfitting_curves_by_architecture(results, save_dir="./plots/overfitting_by_arch"):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有架构和数据集的组合
    architectures = set()
    datasets = set()
    for algo_data in results.values():
        for arch_data in algo_data.items():
            architectures.add(arch_data[0])
            datasets.update(arch_data[1].keys())
    
    # 为每个架构-数据集组合创建单独的图
    for arch in architectures:
        for dataset in datasets:
            plt.figure(figsize=(10, 6))
            
            # 收集所有算法的数据点
            all_in_dist = []
            all_ood = []
            
            # 为每个算法收集数据点（除了bohb）
            for algo, algo_data in results.items():
                if algo == 'bohb' or arch not in algo_data or dataset not in algo_data[arch]:
                    continue
                    
                metrics = algo_data[arch][dataset]
                in_dist = metrics['in_distribution_acc']
                ood = metrics['ood_generalization']
                
                # 计算截止目前为止的最大值
                max_in_dist = []
                corresponding_ood = []
                current_max = float('-inf')
                
                # 计算完整序列
                for i in range(len(in_dist)):
                    if in_dist[i] > current_max:
                        current_max = in_dist[i]
                    max_in_dist.append(current_max)
                    corresponding_ood.append(ood[i])
                
                # 只保留起始值、最大值和最终值
                filtered_points = [(max_in_dist[55], corresponding_ood[55])]  # 起始值
                
                # 找到OOD最大值对应的索引
                max_idx = corresponding_ood.index(max(corresponding_ood))
                filtered_points.append((max_in_dist[max_idx], corresponding_ood[max_idx]))
                
                # 添加最终值
                filtered_points.append((max_in_dist[-1], corresponding_ood[-1]))
                
                all_in_dist.append([p[0] for p in filtered_points])
                all_ood.append([p[1] for p in filtered_points])
            
            # 只有当收集到数据时才绘图和保存数据
            if all_in_dist:
                # 计算均值和标准差
                mean_in_dist = np.mean(all_in_dist, axis=0)
                mean_ood = np.mean(all_ood, axis=0)
                std_in_dist = np.std(all_in_dist, axis=0)
                std_ood = np.std(all_ood, axis=0)
                
                # 保存数据到txt文件
                data_save_path = os.path.join(save_dir, f'data_{arch}_{dataset}.txt')
                with open(data_save_path, 'w') as f:
                    f.write('x y yl yu\n\n')
                    
                    for i in range(len(mean_in_dist)):
                        f.write(f'{mean_in_dist[i]:.4f} {mean_ood[i]:.4f} {mean_ood[i]-std_ood[i]:.4f} {mean_ood[i]+std_ood[i]:.4f}\n')
                
                # 绘制均值曲线
                plt.plot(mean_in_dist, mean_ood, marker='o', color='blue', label='Mean', markersize=4)
                
                # 绘制标准差区域
                plt.fill_between(mean_in_dist, 
                               mean_ood - std_ood,
                               mean_ood + std_ood,
                               alpha=0.2,
                               color='blue',
                               label='±1 std')
                
                plt.xlabel('In-distribution Accuracy')
                plt.ylabel('OOD Generalization')
                plt.title(f'Overfitting Analysis - {arch} - {dataset}')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                
                # 保存图片
                save_path = os.path.join(save_dir, f'overfitting_{arch}_{dataset}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
def calculate_performance_changes(results_dict):
    # 按算法组织数据
    algo_changes = {}
    
    # 遍历所有算法（除了bohb）
    for algo, algo_data in results_dict.items():
        if algo == 'bohb':
            continue
            
        algo_changes[algo] = []
            
        # 遍历架构和数据集
        for arch, arch_data in algo_data.items():
            for dataset, metrics in arch_data.items():
                # 跳过ColoredMNIST数据集
                if dataset != 'RobCifar10':
                    continue
                    
                in_dist = metrics['in_distribution_acc']
                ood = metrics['ood_generalization']
                
                if not in_dist or not ood:  # 跳过空数据
                    continue
                
                # 计算截至目前为止的最大值
                max_in_dist = []
                corresponding_ood = []
                current_max = float('-inf')
                
                for i in range(len(in_dist)):
                    if in_dist[i] > current_max:
                        current_max = in_dist[i]
                    max_in_dist.append(current_max)
                    corresponding_ood.append(ood[i])
                
                # 计算变化
                max_ood_index = corresponding_ood.index(max(corresponding_ood))  # 找到ood最大值的索引
                initial_in_dist = max_in_dist[max_ood_index]  # 使用ood最大值对应的in_dist作为初始值
                final_in_dist = max_in_dist[-1]  # 使用最大值作为最终值
                in_dist_change = ((final_in_dist - initial_in_dist) / initial_in_dist) * 100
                
                initial_ood = max(corresponding_ood)
                final_ood = corresponding_ood[-1]  # 使用最后一个值
                ood_change = ((final_ood - initial_ood) / initial_ood) * 100
                
                algo_changes[algo].append({
                    'arch': arch,
                    'in_dist_change': in_dist_change,
                    'ood_change': ood_change
                })
    
    # 为每个算法保存单独的文件
    output_dir = "./plots/overfitting"
    os.makedirs(output_dir, exist_ok=True)
    
    for algo, changes in algo_changes.items():
        output_file = os.path.join(output_dir, f"{algo}_performance_changes.txt")
        with open(output_file, 'w') as f:
            for change in changes:
                f.write(f"{change['arch']} {change['in_dist_change']:+.3f} {change['ood_change']:+.3f}\n")

# 使用示例
if __name__ == "__main__":
    results = process_exp_data()
    # plot_overfitting_curves(results)
    # plot_overfitting_curves_by_architecture(results)
    calculate_performance_changes(results)


