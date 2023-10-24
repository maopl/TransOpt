import pickle
import numpy as np
from collections import defaultdict
import scipy
import seaborn
import matplotlib.pyplot as plt
import pandas as pds
import os
import seaborn as sns
from sklearn.cluster import DBSCAN
from cliffs_delta import cliffs_delta
from scipy import stats
from ResultAnalysis.heatmap import plot_heatmap
from scipy.stats import norm
from Util.sk import Rx
from collections import Counter







def dbscan_analysis(data):
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



class Res_data():
    def __init__(self):
        self.X = None
        self.Y = None
        self.best_X = None
        self.best_Y = None



class Exp_result_analysis():
    def __init__(self, Exper_floder, Method_list, Dim, Seed_list, test_problem_list, init_point, end_point):
        self._Exper_floder = Exper_floder
        self._Method_list = Method_list
        self._Dim = Dim
        self._Seed_list = Seed_list
        self._init_point =  init_point
        self._end_point = end_point

        self._test_problem_list = []
        for problems in test_problem_list:
            if problems.split('_')[0] == 'XGB' or problems.split('_')[0] == 'SVM' or problems.split('_')[0] == 'NN':
                num = int(problems.split('_')[1])
                for i in range(num):
                    prob_name = problems.split('_')[0]+f'{self._Dim}d' + f'_{i}'
                    self._test_problem_list.append(prob_name)
            elif problems.split('_')[0] == 'lunar':
                num = int(problems.split('_')[1])
                for i in range(num):
                    prob_name = 'Lunar_lander_'+f'{self._Dim}d' + f'_{i}'
                    self._test_problem_list.append(prob_name)
            elif problems.split('_')[0] == 'HPOb':
                num = int(problems.split('_')[1])
                for i in range(num):
                    prob_name = 'HPOb_'+f'{self._Dim}d' + f'_{aa[i]}'
                    self._test_problem_list.append(prob_name)
            elif problems.split('_')[0] == 'NN4d':
                num = int(problems.split('_')[1])
                for i in range(num):
                    prob_name = 'NN4d_'+f'{self._Dim}d'
                    self._test_problem_list.append(prob_name)
            elif problems.split('_')[0] == 'RES':
                num = int(problems.split('_')[1])
                for i in range(num):
                    prob_name = 'RES_'+f'{i}'
                    self._test_problem_list.append(prob_name)
            else:
                num = int(problems.split('_')[1])
                for i in range(num):
                    prob_name  = problems.split('_')[0] + f'_{i}_' + problems.split('_')[2]
                    self._test_problem_list.append(prob_name)

        self._exp_res_data = defaultdict(dict)
        self._exp_res_time = defaultdict(dict)

        self._data_mean = defaultdict(list)
        self._data_std = defaultdict(list)

    def read_data_from_file(self):
        for Method in self._Method_list:
            self._exp_res_data[Method] = defaultdict(list)
            for test_problem in self._test_problem_list:
                self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'] = {}
                for Seed in self._Seed_list:
                    result = Res_data()
                    result.X = np.loadtxt('{}/data/{}/{}d/{}/{}_x.txt'.format(self._Exper_floder, Method, self._Dim, Seed, test_problem))
                    result.Y = np.loadtxt('{}/data/{}/{}d/{}/{}_y.txt'.format(self._Exper_floder, Method, self._Dim, Seed, test_problem))
                    if len(result.Y) < self._end_point:
                        continue
                    best_id = np.argmin(result.Y)
                    result.best_Y = result.Y[best_id]
                    result.best_X = result.X[best_id]

                    self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][Seed] = result

    def pickle_dump_data(self, file_path= 'exp_res_data.pkl'):
        with open('{}/{}'.format(self._Exper_floder, file_path), 'wb') as f:
            pickle.dump(self._exp_res_data, f)


    def print_best(self):
        Rm = []
        result2 = {}
        m_num = len(self._Method_list)
        for Method in self._Method_list:
            result = []
            result2[Method] = []
            for test_problem in self._test_problem_list:
                best = []
                for s_id, s in enumerate(self._Seed_list):
                    try:
                        best.append(np.min(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y[:self._end_point]))
                    except:
                        print(1)

                result.append(np.mean(best))
            Rm.append(result)
        Rm = np.array(Rm).T

        sorted_indices = np.argsort(Rm, axis=1)
        rank_sorted = np.zeros_like(Rm)
        for i in range(Rm.shape[0]):
            rank_sorted[i] = sorted_indices[i].argsort()
        count_per_column = np.zeros((m_num, m_num))
        for m_id, m in enumerate(self._Method_list):
            count_per_column[:, m_id] = np.count_nonzero(rank_sorted == m_id, axis=0)


        for rank in range(m_num):
            print(f'--------------Rank {rank}-------------\n')
            for m_id, m in enumerate(self._Method_list):
                print(f"{m}: {count_per_column[m_id][rank]}")

        b = np.argmin(Rm, axis=1)
        group_size = 5
        num_groups = len(b) // group_size
        for i in range(num_groups):
            group = b[i * group_size: (i + 1) * group_size]

            # 创建一个空字典用于存储每个值的重复次数
            count_dict = {}

            # 遍历组内的每个元素
            for item in group:
                # 如果字典中已经存在该元素，则将其对应的值加 1
                if item in count_dict:
                    count_dict[item] += 1
                # 如果字典中不存在该元素，则将其初始化为 1
                else:
                    count_dict[item] = 1

            # 打印每个值及其对应的重复次数

    def plot_convergence_rate(self, Exper_folder):
        fig = plt.figure(figsize=(14, 9))
        cr_results = {}
        for tid, test_problem in enumerate(self._test_problem_list):
            result = {}
            for mid, Method in enumerate(self._Method_list):
                cr_list = []
                for s_id, s in enumerate(self._Seed_list):
                    cr = convergence_rate(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y, self._init_point-1, self._end_point)
                    cr_list.append(cr)

                result[Method] = cr_list

            a = Rx.data(**result)
            RES = Rx.sk(a)
            for r in RES:
                if r.rx  in cr_results:
                    cr_results[r.rx].append(r.rank)
                else:
                    cr_results[r.rx] = [r.rank]

        df = pds.DataFrame(cr_results)

        # 绘制 violin plot
        sns.violinplot(data=df, inner="quart")

        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)

        plt.savefig('{}/figs/analysis/All/{}.png'.format(Exper_folder, 'convergence_rate'),
                    format='png')
        plt.close()


    def get_anchor(self, test_prob):
        best_result = []
        for mid, Method in enumerate(self._Method_list):
            for s_id, s in enumerate(self._Seed_list):
                best_result.append(np.min(self._exp_res_data[Method][f'{self._Dim}d_{test_prob}'][s].Y))

        return np.quantile(best_result, 0.5)

    def plot_acc_iterations(self, Exper_folder):
        fig = plt.figure(figsize=(14, 9))
        cr_results = {}
        for tid, test_problem in enumerate(self._test_problem_list):
            result = {}
            anchor = self.get_anchor(test_problem)
            for mid, Method in enumerate(self._Method_list):
                cr_list = []
                for s_id, s in enumerate(self._Seed_list):
                    cr = acc_iterations(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y, self._init_point - 1, self._end_point, anchor_value=anchor)
                    cr_list.append(cr)

                result[Method] = cr_list

            a = Rx.data(**result)
            RES = Rx.sk(a)
            for r in RES:
                if r.rx  in cr_results:
                    cr_results[r.rx].append(r.rank)
                else:
                    cr_results[r.rx] = [r.rank]

        df = pds.DataFrame(cr_results)

        # 绘制 violin plot
        sns.violinplot(data=df, inner="quart")

        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)

        plt.savefig('{}/figs/analysis/All/{}.png'.format(Exper_folder, 'acc_iterations'),
                    format='png')
        plt.close()

    def plot_escape_c(self, Exper_folder):
        fig = plt.figure(figsize=(14, 9))
        cr_results = {}
        for tid, test_problem in enumerate(self._test_problem_list):
            result = {}
            for mid, Method in enumerate(self._Method_list):
                cr_list = []
                for s_id, s in enumerate(self._Seed_list):
                    cr = escape_c(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y, self._init_point-1, self._end_point)
                    cr_list.append(cr)

                result[Method] = cr_list

            a = Rx.data(**result)
            RES = Rx.sk(a)
            for r in RES:
                if r.rx  in cr_results:
                    cr_results[r.rx].append(r.rank)
                else:
                    cr_results[r.rx] = [r.rank]

        df = pds.DataFrame(cr_results)

        # 绘制 violin plot
        sns.violinplot(data=df, inner="quart")

        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)

        plt.savefig('{}/figs/analysis/All/{}.png'.format(Exper_folder, 'escape ability'),
                    format='png')
        plt.close()


    def plot_sk(self, Exper_folder):
        fig = plt.figure(figsize=(14, 9))
        cr_results = {}
        for tid, test_problem in enumerate(self._test_problem_list):
            result = {}
            for mid, Method in enumerate(self._Method_list):
                cr_list = []
                for s_id, s in enumerate(self._Seed_list):
                    cr = np.min(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y[:self._end_point])
                    cr_list.append(cr)
                result[Method] = cr_list

            a = Rx.data(**result)
            RES = Rx.sk(a)
            for r in RES:
                if r.rx  in cr_results:
                    cr_results[r.rx].append(r.rank)
                else:
                    cr_results[r.rx] = [r.rank]

        df = pds.DataFrame(cr_results)

        # 绘制 violin plot
        sns.violinplot(data=df, inner="quart")

        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)

        plt.savefig('{}/figs/analysis/All/{}.png'.format(Exper_folder, 'best'),
                    format='png')
        plt.close()


    def plot_traj(self, Exper_folder, mode = 'median'):
        for test_problem in self._test_problem_list:
            fig, ax = plt.subplots(figsize=(12,6))
            fig.clf()
            for mid, Method in enumerate(self._Method_list):
                res = []
                for s_id, s in enumerate(self._Seed_list):
                    arr = self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y
                    min_values = np.minimum.accumulate(arr)
                    res.append(min_values)

                res_median = np.median(np.array(res),axis=0)
                res_std = np.std(np.array(res),axis=0)
                try:
                    plt.plot(list(range(res_median[:self._init_point-1].shape[0], res_median[:self._end_point].shape[0])), res_median[self._init_point-1:self._end_point], label=Method,color=colors[mid])
                    plt.fill_between(list(range(res_median[:self._init_point-1].shape[0], res_median[:self._end_point].shape[0])), res_median[self._init_point-1:self._end_point] + res_std[self._init_point-1:self._end_point], res_median[self._init_point-1:self._end_point] - res_std[self._init_point-1:self._end_point], alpha=0.3,
                                    color=colors[mid])
                except:
                    print(f'Method:{Method}, test_problem:{test_problem}')
                    raise
            plt.axvline(x=self._init_point, color='red', linestyle='--')

            plt.title('Optimization Traj')
            plt.xlabel('Function Evaluations')
            plt.ylabel('Best Result So Far')
            # if test_problem.split('_')[0] == 'DixonPrice':
            #     plt.ylim(-100,100)
            plt.legend(loc='upper left',bbox_to_anchor=(1,1), prop={'size':6.5})

            os.makedirs('{}/figs/analysis/Traj'.format(Exper_folder), exist_ok=True)

            plt.savefig('{}/figs/analysis/Traj/{}.png'.format(Exper_folder, test_problem),
                        format='png')
            plt.close()

    # def plot_violin(self, Exper_folder, mode ='median'):
    #     # fig = plt.figure(figsize=(12, 6))
    #     for test_problem in self._test_problem_list:
    #         fig, ax = plt.subplots(figsize=(12,6))
    #         fig.clf()
    #         data = {'Method':[], 'value':[]}
    #         for mid, Method in enumerate(self._Method_list):
    #             for s_id, s in enumerate(self._Seed_list):
    #                 arr = np.min(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s_id].Y[:self._end_point])
    #                 data['Method'].append(Method)
    #                 data['value'].append(arr)
    #
    #         sns.violinplot(data=data, x='Method', y='value')
    #
    #         os.makedirs('{}/figs/analysis/Violin'.format(Exper_folder), exist_ok=True)
    #
    #         plt.savefig('{}/figs/analysis/Violin/{}.png'.format(Exper_folder, test_problem),
    #                     format='png')
    #         plt.close()




    def plot_violin_all(self, Exper_folder, mode='median'):
        data = {'Method': [], 'value': []}
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.clf()
        for test_problem in self._test_problem_list:
            for s_id, s in enumerate(self._Seed_list):
                value = {}
                for mid, Method in enumerate(self._Method_list):
                        best_output = np.min(
                            self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y[:self._end_point])
                        value[Method] = best_output

                sorted_value = sorted(value.values())
                for v_id,v in enumerate(sorted_value):
                    for k, vv in value.items():
                        if v == vv:
                            data['Method'].append(k)
                            data['value'].append(v_id)

        ax = sns.violinplot(data=data, x='Method', y='value', palette=colors_rgb[:len(self._Method_list)], width=0.5)

        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)
        plt.savefig('{}/figs/analysis/All/{}'.format(Exper_folder, 'violin'),
                    format='png')
        plt.close()

    def box_plot_all(self, Exper_folder, mode='median'):
        Rm = []
        result2 = {}
        m_num = len(self._Method_list)
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.clf()
        for Method in self._Method_list:
            result = []
            result2[Method] = []
            for test_problem in self._test_problem_list:
                best = []
                for s_id, s in enumerate(self._Seed_list):
                    best.append(np.min(
                        self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y[:self._end_point]))

                if mode == 'median':
                    result.append(np.median(best))
                elif mode == 'mean':
                    result.append(np.mean(best))
            Rm.append(result)
        Rm = np.array(Rm).T

        # Calculate ranks using scipy.stats.rankdata

        ranks = np.array([scipy.stats.rankdata(x, method='min') for x in Rm])

        df = pds.DataFrame(ranks, columns=self._Method_list)
        sns.boxplot(df)
        plt.title('Box plot of Ablation')
        plt.xlabel('Algorithm Name')
        plt.ylabel('Rank')

        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)

        plt.savefig('{}/figs/analysis/All/{}'.format(Exper_folder, 'Box'),
                    format='png')
        plt.close()


    def record_mean_std(self):
        res_mean = {}
        res_std = {}
        rst = {}
        for test_problem in self._test_problem_list:
            result_mean = []
            result_std = []
            data = {}
            data_mean ={}
            for Method in self._Method_list:
                best = []
                for s_id, s in enumerate(self._Seed_list):
                    best.append(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].best_Y)

                data[Method] = best.copy()
                data_mean[Method] = (np.mean(best), np.std(best))
                result_mean.append(np.mean(best))
                result_std.append(np.std(best))

            res_mean[test_problem] = result_mean
            res_std[test_problem] = result_std

            rst_m = {}
            sorted_dic = sorted(data_mean.items(), key=lambda kv: (kv[1][0]))
            for Method in self._Method_list:
                if Method == sorted_dic[0][0]:
                    rst_m[Method] = '-'
                    continue
                s, p = stats.mannwhitneyu(data[sorted_dic[0][0]], data[Method],alternative='two-sided')
                if p < 0.05:
                    rst_m[Method] = '+'
                else:
                    rst_m[Method] = '-'
            rst[test_problem] = rst_m
        my_str = matrix_to_latex(res_mean, res_std, rst, self._test_problem_list, self._Method_list)
        print(my_str)



    def record_acc_iterations(self, order='max'):
        res_mean = {}
        res_std = {}
        rst = {}
        for test_problem in self._test_problem_list:
            result_mean = []
            result_std = []
            data = {}
            data_mean ={}
            anchor = self.get_anchor(test_problem)
            for Method in self._Method_list:
                best = []
                for s_id, s in enumerate(self._Seed_list):
                    cr = acc_iterations(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y, self._init_point - 1, self._end_point, anchor_value=anchor)
                    best.append(cr)

                data[Method] = best.copy()
                data_mean[Method] = (np.mean(best), np.std(best))
                result_mean.append(np.mean(best))
                result_std.append(np.std(best))

            res_mean[test_problem] = result_mean
            res_std[test_problem] = result_std

            rst_m = {}
            sorted_dic = sorted(data_mean.items(), key=lambda kv: (kv[1][0]),reverse=False)
            for Method in self._Method_list:
                if Method == sorted_dic[0][0]:
                    rst_m[Method] = '-'
                    continue
                s, p = stats.mannwhitneyu(data[sorted_dic[0][0]], data[Method],alternative='two-sided')
                if p < 0.05:
                    rst_m[Method] = '+'
                else:
                    rst_m[Method] = '-'
            rst[test_problem] = rst_m
        my_str = matrix_to_latex(res_mean, res_std, rst, self._test_problem_list, self._Method_list, oder='min')
        print(my_str)

    def record_convergence(self, order='max'):
        res_mean = {}
        res_std = {}
        rst = {}
        for test_problem in self._test_problem_list:
            result_mean = []
            result_std = []
            data = {}
            data_mean ={}
            anchor = self.get_anchor(test_problem)
            for Method in self._Method_list:
                best = []
                for s_id, s in enumerate(self._Seed_list):
                    cr = convergence_rate(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y, self._init_point - 1, self._end_point)
                    best.append(cr)

                data[Method] = best.copy()
                data_mean[Method] = (np.mean(best), np.std(best))
                result_mean.append(np.mean(best))
                result_std.append(np.std(best))

            res_mean[test_problem] = result_mean
            res_std[test_problem] = result_std

            rst_m = {}
            sorted_dic = sorted(data_mean.items(), key=lambda kv: (kv[1][0]),reverse=True)
            for Method in self._Method_list:
                if Method == sorted_dic[0][0]:
                    rst_m[Method] = '-'
                    continue
                s, p = stats.mannwhitneyu(data[sorted_dic[0][0]], data[Method],alternative='two-sided')
                if p < 0.05:
                    rst_m[Method] = '+'
                else:
                    rst_m[Method] = '-'
            rst[test_problem] = rst_m
        my_str = matrix_to_latex(res_mean, res_std, rst, self._test_problem_list, self._Method_list, oder='min')
        print(my_str)

    def record_escape(self):
        res_mean = {}
        res_std = {}
        rst = {}
        for test_problem in self._test_problem_list:
            result_mean = []
            result_std = []
            data = {}
            data_mean ={}
            for Method in self._Method_list:
                best = []
                for s_id, s in enumerate(self._Seed_list):
                    cr = escape_c(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].Y, self._init_point-1, self._end_point)
                    best.append(cr)

                data[Method] = best.copy()
                data_mean[Method] = (np.mean(best), np.std(best))
                result_mean.append(np.mean(best))
                result_std.append(np.std(best))

            res_mean[test_problem] = result_mean
            res_std[test_problem] = result_std

            rst_m = {}
            sorted_dic = sorted(data_mean.items(), key=lambda kv: (kv[1][0]))
            for Method in self._Method_list:
                if Method == sorted_dic[0][0]:
                    rst_m[Method] = '-'
                    continue
                s, p = stats.mannwhitneyu(data[sorted_dic[0][0]], data[Method],alternative='two-sided')
                if p < 0.05:
                    rst_m[Method] = '+'
                else:
                    rst_m[Method] = '-'
            rst[test_problem] = rst_m
        my_str = matrix_to_latex(res_mean, res_std, rst, self._test_problem_list, self._Method_list)

    def dbscan_analysis(self, Exper_folder):
        nsnscores_d = {}
        ncscores_d = {}
        mu = 0.8  # 假设最优簇大小为所有点数除以两倍的簇的数量
        sigma = 1  # 假设标准差为 mu 的三分之一，您可以根据您的需求来调整这个值
        for test_problem in self._test_problem_list:
            cln_ = {}
            nsn_ = {}
            cs_ = {}
            nsnscore_ = {}
            ncscore_ = {}
            for Method in self._Method_list:
                cln_l = []
                nsn_l = []
                cs_l = []
                nsnscore_l = []
                ncscore_l = []
                for s_id, s in enumerate(self._Seed_list):
                    cln, nsn, cs = dbscan_analysis(self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].X)

                    cln_l.append(cln)
                    nsn_l.append(nsn)
                    cs_l.append(cs)
                    nsnscore =  nsn / self._exp_res_data[Method][f'{self._Dim}d_{test_problem}'][s].X.shape[0]

                    # 计算每个簇的 Gaussian Score 的平均值
                    # nsnscore = norm.pdf(nsnscore, loc=mu, scale=sigma)

                    ncscore = 2 / (1 + np.exp(-0.5 * cln)) - 1
                    nsnscore_l.append(nsnscore)
                    ncscore_l.append(ncscore)

                nsn_[Method] = nsn_l
                cln_[Method] = cln_l
                cs_[Method] = cs_l
                nsnscore_[Method] = nsnscore_l
                ncscore_[Method] = ncscore_l
            nsnscores_d[test_problem] = nsnscore_
            ncscores_d[test_problem] = ncscore_

            # 创建1行2列的子图
            fig, axs = plt.subplots(1, 3, figsize=(12, 5))

            # 第一个子图显示nsn的数据
            df_nsn = pds.DataFrame(nsn_, columns=self._Method_list)
            sns.boxplot(ax=axs[0], data=df_nsn)
            axs[0].set_ylabel('NSn_Performance')
            axs[0].set_xlabel('Algorithm')
            axs[0].set_title('NSn Performance Comparison')
            axs[0].yaxis.grid(True)

            # 第二个子图显示cln的数据
            df_cln = pds.DataFrame(cln_, columns=self._Method_list)
            sns.boxplot(ax=axs[1], data=df_cln)
            axs[1].set_ylabel('CLN_Performance')
            axs[1].set_xlabel('Algorithm')
            axs[1].set_title('CLn Performance Comparison')
            axs[1].yaxis.grid(True)

            # 第二个子图显示cln的数据
            df_cln = pds.DataFrame(cs_, columns=self._Method_list)
            sns.boxplot(ax=axs[2], data=df_cln)
            axs[2].set_ylabel('Cluster_Size_Performance')
            axs[2].set_xlabel('Algorithm')
            axs[2].set_title('Cluster_size Performance Comparison')
            axs[2].yaxis.grid(True)


            plt.tight_layout()

            os.makedirs('{}/figs/analysis/dba'.format(Exper_folder), exist_ok=True)


            plt.savefig('{}/figs/analysis/dba/{}'.format(Exper_folder, f'{test_problem}'),
                        format='png')
            plt.close()

        data = {'Method': [], 'value': []}
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.clf()
        for Method in self._Method_list:
            for test_problem in self._test_problem_list:
                for v_id in range(len(ncscores_d[test_problem][Method])):

                    data['Method'].append(Method)
                    data['value'].append(nsnscores_d[test_problem][Method][v_id])

        ax = sns.boxplot(data=data, x='Method', y='value', palette=colors_rgb[:len(self._Method_list)], width=0.5)
        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)
        plt.savefig('{}/figs/analysis/All/{}'.format(Exper_folder, 'dbscan exploration scores'),
                    format='png')


        data = {'Method': [], 'value': []}
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.clf()
        for Method in self._Method_list:
            for test_problem in self._test_problem_list:
                for v_id in range(len(ncscores_d[test_problem][Method])):

                    data['Method'].append(Method)
                    data['value'].append(ncscores_d[test_problem][Method][v_id])

        ax = sns.boxplot(data=data, x='Method', y='value', palette=colors_rgb[:len(self._Method_list)], width=0.5)
        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)
        plt.savefig('{}/figs/analysis/All/{}'.format(Exper_folder, 'dbscan exploitation scores'),
                    format='png')

        data = {'Method': [], 'value': []}
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.clf()
        for Method in self._Method_list:
            for test_problem in self._test_problem_list:
                for v_id in range(len(ncscores_d[test_problem][Method])):

                    data['Method'].append(Method)
                    data['value'].append(ncscores_d[test_problem][Method][v_id] * nsnscores_d[test_problem][Method][v_id])

        ax = sns.boxplot(data=data, x='Method', y='value', palette=colors_rgb[:len(self._Method_list)], width=0.5)
        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)
        plt.savefig('{}/figs/analysis/All/{}'.format(Exper_folder, 'balance scores'),
                    format='png')

aa = ['9983', '31', '37', '3902', '9977', '125923']
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
colors_rgb = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

if __name__ == '__main__':
    task_list_2d = [
        # 'Ackley_3_s',
        # 'MPB5_3_s',
        # 'Griewank_3_s',
        # 'DixonPrice_3_s',
        # 'Rosenbrock_3_s',
        # 'RotatedHyperEllipsoid_3_s',
        'SVM_3_s',
    ]
    task_list_4d = [
        # 'NN_2_s',
        'RES_3_s',
    ]

    task_list_8d = [
        # 'Ackley_5_s',
        # 'MPB5_5_s',
        # 'Griewank_5_s',
        # 'DixonPrice_5_s',
        # 'Rosenbrock_5_s',
        # 'RotatedHyperEllipsoid_5_s',
        'XGB_5_s',
    ]

    task_list_10d = [
        'Ackley_3_s',
        'MPB5_3_s',
        'Griewank_3_s',
        'DixonPrice_3_s',
        'Rosenbrock_3_s',
        'RotatedHyperEllipsoid_3_s',
        'lunar_3_s',
        'XGB_3_s',
    ]

    Dim_ = 10
    Method_list = [
        'LFLBO',
        'MHGP',
        # 'TST',
        'MTBO',
        'HyperBO',
        'BO',
        # 'abl1',
        # 'abl2',

    ]
    # Seed_list = range(1,10)
    Seed_list = [1,2,3,4,5,6,7,8,9,10]

    # Seed_list = [12,8,9,10]
    Exp_name = 'Results'
    Exper_floder = '../../LFL_experiments/{}'.format(Exp_name)

    if Dim_ == 2:
        task_list = task_list_2d
    elif Dim_ == 4:
        task_list = task_list_4d
    elif Dim_ == 8:
        task_list = task_list_8d
    elif Dim_ == 10:
        task_list = task_list_10d
    init_point = 4 *Dim_
    end_point = 10*Dim_
    Exp_res = Exp_result_analysis(Exper_floder, Method_list, Dim_, Seed_list, task_list, init_point=init_point, end_point=end_point)
    Exp_res.read_data_from_file()
    Exp_res.print_best()
    # Exp_res.plot_sk(Exper_floder)
    # Exp_res.record_mean_std()
    # Exp_res.plot_acc_iterations(Exper_floder)
    # Exp_res.record_acc_iterations()
    # Exp_res.plot_convergence_rate(Exper_floder)
    Exp_res.record_convergence()
    # Exp_res.plot_escape_c(Exper_floder)
    # Exp_res.dbscan_analysis(Exper_floder)
    # Exp_res.plot_violin_all(Exper_floder)
    # Exp_res.plot_traj(Exper_floder)
    # Exp_res.box_plot_all(Exper_floder, 'median')






