import math
import logging
import random

import numpy as np
stand = 1
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

from Benchmark.Synthetic.SyntheticBenchmark import BASE
from KnowledgeBase.CKB import KnowledgeBase
from Gym import Gym_Metric
from sklearn.neighbors import KernelDensity

def construct_setting(X, Y, match_id):
    trend_perf = Gym_Metric.trend_metric(X, Y)
    # rug_perf = np.mean(Gym_Metric.rug_metric(X, Y))
    g_optima = X[np.argmin(Y)].reshape(1,len(X[0]))
    Gym_setting = {'match_id': match_id, 'g_optima':g_optima}

    return Gym_setting


class Gym_c:
    def __init__(self, X, Y, n):
        self.X = X
        self.Y = Y
        self.n_clusters = n
        self.kde_models = []
        self._cluster_data()
        self._fit_kde()

    def _cluster_data(self):
        """
        对Y进行聚类，并将X按照Y的聚类结果分类。
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.Y.reshape(-1, 1))
        labels = kmeans.labels_
        self.clustered_data = {}
        for i in range(self.n_clusters):
            self.clustered_data[i] = self.X[labels == i]

    def _fit_kde(self):
        """
        对每个簇的X数据使用KDE拟合。
        """
        for key, X_cluster in self.clustered_data.items():
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_cluster)
            self.kde_models.append(kde)

    def sampling_probability(self, sample_x):
        """
        对于给定的x样本，输出每个KDE的采样概率。
        """
        probabilities = {}
        for i, kde in enumerate(self.kde_models):
            log_prob = kde.score_samples([sample_x])
            probabilities[i] = np.exp(log_prob)[0]
        return probabilities





class Gym(BASE):
    def __init__(
            self,
            fun_name,
            Gym_setting,
            input_dim=1,
            sd=None,
            Seed=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0

        self.var_bound = np.array([[0, 1]] * input_dim)
        bounds = np.array([[-1.0] * input_dim, [1.0] * input_dim], dtype=np.float64) * stand


        optimizers = tuple([0] * input_dim)
        super(Gym, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=self.var_bound,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )

        self.g_height_bound = np.array([[0.3, 0.7]] * 1)
        self.g_width_bound = np.array([[0.1, 0.8]] * 1)
        if 'g_optima' in Gym_setting:
            self.global_optima = Gym_setting['g_optima']
        else:
            self.global_optima = np.random.uniform(low=0, high=1, size=(1, input_dim)) \
                                  * np.tile(self.var_bound[:, 1] - self.var_bound[:, 0], (1, 1)) \
                                  + np.tile(self.var_bound[:, 0], (1, 1))
        if 'g_width' in Gym_setting:
            self.global_width = Gym_setting['g_width']
        else:
            self.global_width = np.random.random(size=(1,)) \
                                 * (self.g_width_bound[:, 1] - self.g_width_bound[:, 0]) \
                                 + self.g_width_bound[:, 0]
        if 'g_height' in Gym_setting:
            self.global_height = Gym_setting['g_height']
        else:
            self.global_height = np.random.random(size=(1,)) \
                                 * (self.g_height_bound[:, 1] - self.g_height_bound[:, 0]) \
                                 + self.g_height_bound[:, 0]

        if 'n_peak' in Gym_setting:

            self.n_peak = Gym_setting['n_peak']
            if self.n_peak > 0:
                self.local_height_bound = np.array([[0.2, 0.5]] * self.n_peak)
                self.local_width_bound = np.array([[0.1, 0.3]] * self.n_peak)

                self.local_peak = np.random.uniform(low=0,high=1,size=(self.n_peak, input_dim)) \
                                        * np.tile(self.var_bound[:, 1] - self.var_bound[:, 0], (self.n_peak, 1)) \
                                        + np.tile(self.var_bound[:, 0], (self.n_peak, 1))

                self.current_width = np.random.random(size=(self.n_peak,)) \
                                     * (self.local_width_bound[:, 1] - self.local_width_bound[:, 0]) \
                                     + self.local_width_bound[:, 0]

                self.current_height = np.random.random(size=(self.n_peak,)) \
                                      * (self.local_height_bound[:, 1] - self.local_height_bound[:, 0]) \
                                      + self.local_height_bound[:, 0]
        else:
            self.n_peak = 0

        if 'rugdness' in Gym_setting:
            self.rug = Gym_setting['rugdness']
        else:
            # self.rug = np.random.uniform(0,1)
            self.rug = 0.1


        if 'noise' in Gym_setting:
            self.noise = Gym_setting['noise']
        else:
            pass

        if 'beta' in Gym_setting:
            self.beta = Gym_setting['beta']
        else:
            self.beta = 0.5

        if 'match_id' in Gym_setting:
            self.match_id = Gym_setting['match_id']
        else:
            self.match_id = -1
    def map_to_discrete_space(self, matrix):
        interval = 2*self.beta / (2 * self.beta + 1)
        discrete_matrix = np.round(matrix / interval) * interval

        return discrete_matrix


    def peak_function_cone(self,x):
        if self.n_peak != 0:
            current_peak = np.concatenate((self.local_peak, self.global_optima))
            current_height =  np.concatenate((self.current_height, self.global_height))
            current_width =  np.concatenate((self.current_width, self.global_width))

            distance = np.linalg.norm(np.tile(x, (1+self.n_peak, 1)) - current_peak, axis=1)
        else:
            current_peak = self.global_optima
            current_height = self.global_height
            current_width = self.global_width

            distance = np.linalg.norm(np.tile(x,(1,1)) - current_peak,axis=1)

        return np.max(current_height - current_width * distance)
    def f(self, X, peak_shape = 'cone'):
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))
        self.query_num += 1
        X = self.map_to_discrete_space(X)
        X = self.transfer(X)
        n_sample = X.shape[0]
        part1 = np.zeros(shape=(n_sample, 1))

        for i in range(n_sample):
            part1[i, 0] = self.peak_function_cone(X[i, :self.xdim])
        part1 = -part1.reshape((part1.shape[0],))
        part2 = 0.5 * np.sum(np.sin(self.rug *20 * X), axis=1) / self.xdim
        part3 = self.noise(n_sample)

        return part1 + part2 + part3




#
# class Gym(BASE):
#     def __init__(
#             self,
#             fun_name,
#             match_id,
#             Gym_setting,
#             input_dim=1,
#             bounds=None,
#             sd=None,
#             RX=None,
#             Seed=0,
#             shift=None,
#             stretch = None,
#             dtype=np.float64,
#     ):
#         self.name = fun_name
#         self.optimal_value = 0.0
#         if shift is None:
#             self.shift = np.array([0]*input_dim, dtype=np.float64)
#         else:
#             self.shift = shift
#         if stretch is None:
#             self.stretch = np.array([1] * input_dim, dtype=np.float64)
#         else:
#             self.stretch = stretch
#
#         self.noise_freq = Gym_setting['noise_freq']
#         self.noise_amplitude = Gym_setting['noise_ampltude']
#         self.trend = Gym_setting['trend']
#         self.rug = Gym_setting['rug']
#
#         self.noise_freq = 10
#         self.noise_amplitude = 0.12
#         self.trend = 1
#         self.rug = 7
#
#         self.match_id = match_id
#
#         RX = [(-5, 5) for _ in range(input_dim)] if RX is None else RX
#         optimizers = tuple(self.shift)
#         super(Gym, self).__init__(
#             input_dim=input_dim,
#             bounds=bounds,
#             sd=sd,
#             RX=RX,
#             optimizers=optimizers,
#             Seed=Seed,
#             dtype=dtype,
#         )
#
#
#     def f(self, X):
#         if len(X.shape) == 1:
#             X = X.reshape(shape=(1, self.xdim))
#         self.query_num += 1
#         X = self.stretch * (X - self.shift)
#         X = self.transfer(X)
#
#         ##control the noise
#         part1 = self.noise_amplitude * np.exp(np.sum(np.cos(X*self.noise_freq), axis=1) * (1/self.xdim))
#
#         ###control the global optima and the trend (1~3)
#         part2 = np.sqrt(np.sum((self.trend*X)**2, axis=1) * (1 / self.xdim))
#
#         ##control the rugness (0.5~3)
#         part3 = X[:,0] * np.sin(self.rug * X[:,0])
#
#
#         return part2 + part1 + part3



def plot_true_oned(obj_fun_list, dim, dtype, Exper_floder=None):

    for i in obj_fun_list:
        f, ax = plt.subplots(1, 1, figsize=(16, 6))
        problem = Select_synthetic_fun(fun_name=i, input_dim=dim, Seed=0, dtype=dtype)
        bounds = problem.bounds
        opt_x = problem.optimizers
        opt_val = problem.optimal_value
        test_x = np.arange(-1, 1.05, 0.005, dtype=dtype)
        test_y = problem.f(test_x[:, np.newaxis])
        # test_y = Normalize(test_y)
        ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
        ax.legend(['True f(x)'])
        ax.set_xlim([bounds[0][0], bounds[1][0]])
        ax.set_title(i)
        # plt.show()
        if not os.path.exists('{}/true_f/oneD/'.format(Exper_floder)):
            os.makedirs('{}/true_f/oneD/'.format(Exper_floder))
        name = problem.name
        if '.' in problem.name:
            name = name.replace('.','|')

        save_load = '{}/true_f/oneD/{}'.format(Exper_floder, name)

        plt.savefig(save_load+'')

def plot_true_contour(obj_fun_list, Dim, dtype, Exper_floder=None):
    for i in obj_fun_list:
        shift =np.random.random(size=(Dim)) * 0

        stretch = np.random.random(size=(Dim)) * 0.4 + 0.8
        obj_fun = Select_synthetic_fun(fun_name=i, input_dim=Dim, Seed=0, shift=shift, stretch=stretch)

        if not os.path.exists('{}/true_f/contour/'.format(Exper_floder, obj_fun.name)):
            os.makedirs('{}/true_f/contour/'.format(Exper_floder, obj_fun.name))
        name = obj_fun.name
        if '.' in obj_fun.name:
            name = name.replace('.','|')
        save_load = '{}/true_f/contour/{}'.format(Exper_floder, name)

        x = np.linspace(-1, 1, 101)
        y = np.linspace(-1, 1, 101)
        X, Y = np.meshgrid(x, y)
        all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
        Z_true = obj_fun.f(all_sample)
        Z_true = Z_true[:,np.newaxis]
        Z_true = np.asarray(Z_true)
        Z_true = Z_true.reshape(X.shape)

        optimizers = obj_fun.optimizers

        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        a = plt.contourf(X, Y, Z_true, 100, cmap=plt.cm.summer)
        # b = plt.contour(X, Y, Z_true, 50, colors='black', linewidths=1, linestyles='solid')
        # plt.plot(optimizers[:, 0], optimizers[:, 1], marker='*', linewidth=0, color='white', markersize=10, label="GlobalOpt")
        plt.colorbar(a)
        plt.title(i)
        fig.legend(facecolor='gray')
        plt.draw()
        plt.savefig(save_load, dpi=300)
        plt.close()


def plot_true_3D(obj_fun_list, Dim, dtype, Exper_floder=None):
    for i in obj_fun_list:
        a = np.array([[-0.367, 0.958, -0.623, 0.967, 0.5374, 0.728]]).T
        Gym_setting = {'g_optima':a}
        Gym_setting['n_peak'] = 0
        Gym_setting['rugdness'] = 0.1
        Gym_setting['beta'] = 0.01
        obj_fun = Gym(fun_name=i, Gym_setting=Gym_setting, input_dim=Dim, Seed=1)

        fig = plt.figure()  # 定义新的三维坐标轴
        ax = plt.axes(projection='3d')

        # 定义三维数据
        x = np.linspace(-1, 1, 101)
        y = np.linspace(-1, 1, 101)
        X, Y = np.meshgrid(x, y)
        all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
        Z_true = obj_fun.f(all_sample)
        Z_true = Z_true[:,np.newaxis]
        Z_true = np.asarray(Z_true)
        Z = Z_true.reshape(X.shape)


        # 作图
        a = ax.plot_surface(X, Y, Z, cmap=plt.cm.summer)
        # ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
        if not os.path.exists('{}/true_f/3D/'.format(Exper_floder, obj_fun.name)):
            os.makedirs('{}/true_f/3D/'.format(Exper_floder, obj_fun.name))
        name = obj_fun.name
        if '.' in obj_fun.name:
            name = name.replace('.','|')
        save_load = '{}/true_f/3D/{}'.format(Exper_floder, name)
        plt.colorbar(a)
        plt.draw()
        plt.show()
        # print(1)
        # plt.savefig(save_load, dpi=300)


if __name__ == '__main__':
    obj_fun_list = [
        'Gym_0',
    ]
    Dim = 2
    plot_true_3D(obj_fun_list, Dim,  np.float64,'../experiments/plot_problem')