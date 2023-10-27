import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict
from ResultAnalysis.AnalysisBase import AnalysisBase
import matplotlib.pyplot as plt
import pandas as pds
import os
import seaborn as sns
from Util.sk import Rx
from pathlib import Path
import scipy

plot_registry = {}


# 注册函数的装饰器
def metric_register(name):
    def decorator(func_or_class):
        if name in plot_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        plot_registry[name] = func_or_class
        return func_or_class
    return decorator



@metric_register('cr')
def convergence_rate(ab:AnalysisBase, save_path:Path, **kwargs):
    fig = plt.figure(figsize=(14, 9))
    cr_list = []
    cr_all = {}
    cr_results = {}
    def acc_iter(Y, anchor_value):
        for i in range(1, len(Y)):
            best_fn = np.min(Y[:i])
            if best_fn <= anchor_value:
                return i
        return len(Y)

    results = ab.get_results_by_order(["method", "seed", "task"])
    best_Y_values = defaultdict(list)

    # 遍历 data 字典，收集 best_Y 值
    for method, tasks in results.items():
        for seed, task_seed in tasks.items():
            for task_name, result_obj in task_seed.items():
                best_Y = result_obj.best_Y
                if best_Y is not None:
                    best_Y_values[task_name].append(best_Y)

    # 计算并返回每个 task_name 下 best_Y 值的 3/4 分位数
    quantiles = {task_name: np.percentile(values, 75) for task_name, values in best_Y_values.items()}

    for method, tasks in results.items():
        for seed, task_seed in tasks.items():
            for task_name, result_obj in task_seed.items():
                Y = result_obj.Y
                if Y is None:
                    raise ValueError(f"Y is not set for method {method}, task {task_name}")

                cr = acc_iter(Y, anchor_value=quantiles[task_name])
                cr_list.append(cr)

        cr_all[method] = cr_list

    a = Rx.data(**cr_all)
    RES = Rx.sk(a)
    for r in RES:
        if r.rx in cr_results:
            cr_results[r.rx].append(r.rank)
        else:
            cr_results[r.rx] = [r.rank]

    cr_results = pds.DataFrame(cr_results)

    # 绘制 violin plot
    sns.violinplot(data=cr_results, inner="quart")

    os.makedirs(save_path, exist_ok=True)
    save_file = save_path / 'convergence_rate.png'
    plt.savefig(save_file, format='png')
    plt.close()





@metric_register('traj')
def plot_traj(ab, save_path, **kwargs):
    # 先找出所有的任务名称

    results = ab.get_results_by_order(["task", "method", "seed"])

    for task_name, tasks_r in results.items():
        for method, method_r in tasks_r.items():
            res = []
            for seed, result_obj in method_r.items():
                Y = result_obj.Y
                if Y is not None:
                    min_values = np.minimum.accumulate(Y)
                    res.append(min_values)

            res_median = np.median(np.array(res), axis=0)
            res_std = np.std(np.array(res), axis=0)

            plt.plot(list(range(res_median.shape[0])),
                     res_median, label=method, color=ab.get_color_for_method(method))
            plt.fill_between(
                list(range(res_median.shape[0])),
                res_median[:,0] + res_std[:,0], res_median[:,0] - res_std[:,0], alpha=0.3,
                color=ab.get_color_for_method(method))

        plt.title(f'Optimization Trajectory for {task_name}')
        plt.xlabel('Function Evaluations')
        plt.ylabel('Best Result So Far')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 6.5})

        file_path = save_path / f"{task_name}.png"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(file_path, format='png', bbox_inches='tight')
        plt.close()


@metric_register('violin')
def plot_violin(ab:AnalysisBase, save_path, **kwargs):
    data = {'Method': [], 'value': []}



    results = ab.get_results_by_order(["task", "seed", "method"])
    # 为每个任务生成一张图
    plt.figure(figsize=(12, 6))

    for task_name, task_r in results.items():
        for seed, seed_r in task_r.items():
            res = {}
            for method, result_obj in seed_r.items():
                Y = result_obj.Y
                if Y is not None:
                    min_values = np.min(Y)
                    res[method] = min_values
            sorted_value = sorted(res.values())
            for v_id, v in enumerate(sorted_value):
                for k, vv in res.items():
                    if v == vv:
                        data['Method'].append(k)
                        data['value'].append(v_id)


    ax = sns.violinplot(data=data, x='Method', y='value', palette=ab.get_color_for_method(list(ab.get_methods())), width=0.5)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'violin.png', format='png')
    plt.close()



def plot_box(results, save_path, **kwargs):
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'median'
    data = {'Method': [], 'value': []}
    all_seed = set()
    all_task_names = set()
    methods = set()
    for method, tasks in results.items():
        methods.add(method)
        for Seed, task_seed in tasks.items():
            all_seed.add(Seed)
            for task_name in task_seed.keys():
                all_task_names.add(task_name)

    result_list = []
    for method, tasks in results.items():
        result = []
        for task_name in all_task_names:
            best = []
            for Seed in all_seed:
                result_obj = tasks[Seed][task_name]
                if result_obj is not None:
                    Y = result_obj.Y
                    if Y is not None:
                        min_values = np.min(Y)
                        best.append(min_values)
            if mode == 'median':
                result.append(np.median(best))
            elif mode == 'mean':
                result.append(np.mean(best))
        result_list.append(result)
    result_list = np.array(result_list).T


    ranks = np.array([scipy.stats.rankdata(x, method='min') for x in result_list])

    df = pds.DataFrame(ranks, columns=methods)
    sns.boxplot(df)
    plt.title('Box plot of Ablation')
    plt.xlabel('Algorithm Name')
    plt.ylabel('Rank')

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'box.png', format='png')
    plt.close()


# @metric_register('dbscan')
def dbscan_analysis(data, save_path, **kwargs):
    db = DBSCAN(eps=0.5, min_samples=5)
    # 执行聚类
    db.fit(data)

    # 获取聚类标签
    labels = db.labels_

    # 计算簇的数量（忽略噪声点，其标签为 -1）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_sizes = Counter(labels)
    noise_points = cluster_sizes[-1]  # 标签为 -1 的点是噪声点

    # 计算平均簇大小（不包括噪声点）
    if n_clusters > 0:
        t_size = 0
        for ids, cs in cluster_sizes.items():
            if ids >=0:
                t_size += cs
        avg_cluster_size = t_size /n_clusters
    else:
        avg_cluster_size = 0

    return n_clusters, noise_points, avg_cluster_size


# @metric_register('heatmap')
def plot_heatmap(algorithms, test_problems):
    # 创建一个空的矩阵来存储结果
    results_matrix = np.empty((len(test_problems), len(algorithms)))

    # 逐个读取结果文件并填充矩阵
    for i, problem in enumerate(test_problems):
        for j, algorithm in enumerate(algorithms):
            # 从文件读取结果
            file_path = f"{algorithm}_{problem}.txt"
            with open(file_path, "r") as file:
                result = float(file.readline().strip())
            # 填充矩阵
            results_matrix[i, j] = result

    # 创建热力图
    fig, ax = plt.subplots()
    im = ax.imshow(results_matrix, cmap="viridis")

    # 设置轴标签
    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_yticks(np.arange(len(test_problems)))
    ax.set_xticklabels(algorithms)
    ax.set_yticklabels(test_problems)

    # 在热力图上显示数值
    for i in range(len(test_problems)):
        for j in range(len(algorithms)):
            text = ax.text(j, i, f"{results_matrix[i, j]:.2f}",
                           ha="center", va="center", color="w")

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 设置图形标题和标签
    plt.title("Algorithm Comparison Heatmap")
    plt.xlabel("Algorithms")
    plt.ylabel("Test Problems")

    # 显示图形
    plt.show()