import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict
from ResultAnalysis.AnalysisBase import AnalysisBase
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pds
import os
import seaborn as sns
from Util.sk import Rx
from pathlib import Path
import scipy
import tikzplotlib

plot_registry = {}

# 注册函数的装饰器
def metric_register(name):
    def decorator(func_or_class):
        if name in plot_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        plot_registry[name] = func_or_class
        return func_or_class
    return decorator



# @metric_register('cr')
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





# @metric_register('traj')
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
    data = {'Method': [], 'Performance rank': []}
    method_names = set()

    results = ab.get_results_by_order(["task", "seed", "method"])

    for task_name, task_r in results.items():
        for seed, seed_r in task_r.items():
            res = {}
            for method, result_obj in seed_r.items():
                method_names.add(method)
                Y = result_obj.Y
                if Y is not None:
                    min_values = np.min(Y)
                    res[method] = min_values
            sorted_value = sorted(res.values())
            for v_id, v in enumerate(sorted_value):
                for k, vv in res.items():
                    if v == vv:
                        data['Method'].append(k)
                        data['Performance rank'].append(v_id+1)

    sns.set_theme(style="whitegrid", font='FreeSerif')
    plt.figure(figsize=(12, 7.3))
    plt.ylim(bottom=0.9, top=len(method_names)+0.1)
    ax = plt.gca()  # 获取坐标轴对象
    y_major_locator = MultipleLocator(1)  # 设置坐标的主要刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 应用在纵坐标上
    sns.violinplot(x='Method', y='Performance rank', data=data,
                   order=list(method_names),
                   inner="box", color="silver", cut=0, linewidth=3)
    plt.title('Violin plot', fontsize=30, y=1.01)
    plt.xlabel('Algorithm Name', fontsize=25, labelpad=-7)
    plt.ylabel('Performance rank', fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=15)

    save_path = Path(save_path / 'Overview' / 'temp')
    save_path.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(save_path / "violin.tex")
    plt.close()


@metric_register('box')
def plot_box(ab:AnalysisBase, save_path, **kwargs):
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'median'
    data = {'Method': [], 'value': []}
    all_seed = set()
    all_task_names = set()
    methods = set()

    results = ab.get_results_by_order(["method", "task", "seed"])

    result_list = []
    for method, method_r in results.items():
        methods.add(method)
        result = []
        for task, task_r in method_r.items():
            best = []
            for seed, result_obj in task_r.items():
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

    sns.set_theme(style='whitegrid', font='FreeSerif')
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    y_major_locator = MultipleLocator(1)
    ax.yaxis.set_major_locator(y_major_locator)
    sns.boxplot(df, color='#c2d0e9')
    plt.title('Box plot', fontsize=30, y=1.03)
    plt.xlabel('Algorithm Name', fontsize=25, labelpad=-5)
    plt.ylabel('Rank', fontsize=25)
    plt.xticks(fontsize=20, rotation=15)
    plt.yticks(fontsize=20)

    save_path = Path(save_path /'Overview' / 'temp')
    save_path.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(save_path / "box.tex")
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