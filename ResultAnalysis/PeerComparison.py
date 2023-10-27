import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pds
import os
import seaborn as sns
from scipy import stats
from Util.sk import Rx


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

            count_dict = {}
            for item in group:
                if item in count_dict:
                    count_dict[item] += 1
                else:
                    count_dict[item] = 1



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

        sns.violinplot(data=df, inner="quart")

        os.makedirs('{}/figs/analysis/All'.format(Exper_folder), exist_ok=True)

        plt.savefig('{}/figs/analysis/All/{}.png'.format(Exper_folder, 'best'),
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

aa = ['9983', '31', '37', '3902', '9977', '125923']
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
colors_rgb = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

if __name__ == '__main__':
    task_list_2d = [
        # 'Rosenbrock_3_s',
        # 'RotatedHyperEllipsoid_3_s',
        'SVM_3_s',
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
        'MTBO',
        'HyperBO',
        'BO',
    ]
    Seed_list = [1,2,3,4,5,6,7,8,9,10]

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
    # Exp_res.record_acc_iterations()
    Exp_res.record_convergence()







