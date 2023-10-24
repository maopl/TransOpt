import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pds
import os
import seaborn as sns
from Util.sk import Rx
from pathlib import Path
import scipy

metric_registry = {}

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
colors_rgb = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 注册函数的装饰器
def metric_register(name):
    def decorator(func_or_class):
        if name in metric_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        metric_registry[name] = func_or_class
        return func_or_class
    return decorator



@metric_register('cr')
def convergence_rate(results, save_path, **kwargs):
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

    best_Y_values = defaultdict(list)

    # 遍历 data 字典，收集 best_Y 值
    for method, tasks in results.items():
        for Seed, task_seed in tasks.items():
            for task_name, result_obj in task_seed.items():
                best_Y = result_obj.best_Y
                if best_Y is not None:
                    best_Y_values[task_name].append(best_Y)

    # 计算并返回每个 task_name 下 best_Y 值的 3/4 分位数
    quantiles = {task_name: np.percentile(values, 75) for task_name, values in best_Y_values.items()}

    for method, tasks in results.items():
        for Seed, task_seed in tasks.items():
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
    save_file = save_path / 'acc_iterations'
    plt.savefig(save_file, format='png')
    plt.close()




@metric_register('traj')
def plot_traj(results, save_path, **kwargs):
    # 先找出所有的任务名称
    all_task_names = set()
    for method, tasks in results.items():
        for Seed, task_seed in tasks.items():
            for task_name in task_seed.keys():
                all_task_names.add(task_name)

    # 为每个任务生成一张图
    for task_name in all_task_names:
        plt.figure(figsize=(12, 6))
        for method_id, (method, tasks) in enumerate(results.items()):
            res = []

            for Seed, task_seed in tasks.items():
                result_obj = task_seed.get(task_name)
                if result_obj is not None:
                    Y = result_obj.Y
                    if Y is not None:
                        min_values = np.minimum.accumulate(Y)
                        res.append(min_values)

            if not res:
                continue  # 如果这个任务在这个方法下没有结果，跳过

            res_median = np.median(np.array(res), axis=0)
            res_std = np.std(np.array(res), axis=0)

            plt.plot(list(range(res_median.shape[0])),
                     res_median, label=method, color=colors[method_id])
            plt.fill_between(
                list(range(res_median.shape[0])),
                res_median[:,0] + res_std[:,0], res_median[:,0] - res_std[:,0], alpha=0.3,
                color=colors[method_id])

        plt.title(f'Optimization Trajectory for {task_name}')
        plt.xlabel('Function Evaluations')
        plt.ylabel('Best Result So Far')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 6.5})

        file_path = save_path / f"{task_name}.png"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(file_path, format='png', bbox_inches='tight')
        plt.close()


@metric_register('violin')
def plot_violin(results, save_path, **kwargs):
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
    res = {}
    for seed in all_seed:
        res[seed] = {}
    # 为每个任务生成一张图
    for task_name in all_task_names:
        plt.figure(figsize=(12, 6))
        for method_id, (method, tasks) in enumerate(results.items()):
            for Seed, task_seed in tasks.items():
                result_obj = task_seed.get(task_name)
                if result_obj is not None:
                    Y = result_obj.Y
                    if Y is not None:
                        min_values = np.min(Y)
                        res[Seed][method] = min_values

        for Seed in all_seed:
            sorted_value = sorted(res[Seed].values())
            for v_id, v in enumerate(sorted_value):
                for k, vv in res[Seed].items():
                    if v == vv:
                        data['Method'].append(k)
                        data['value'].append(v_id)

    ax = sns.violinplot(data=data, x='Method', y='value', palette=colors_rgb[:len(methods)], width=0.5)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'violin', format='png')
    plt.close()



@metric_register('box')
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
    plt.savefig(save_path / 'box', format='png')
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