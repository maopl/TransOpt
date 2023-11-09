import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, Dict
import ConfigSpace as CS
from Util.Register import benchmark_register, benchmark_registry
from Benchmark.BenchBase.ConfigOptBenchmark import NonTabularOptBenchmark
logger = logging.getLogger('CocoBenchmark')




def plot_true_function(obj_fun_list, Dim, dtype, Exper_folder=None, plot_type="1D"):
    for fun in obj_fun_list:
        obj_fun = get_problem(fun, seed=0, Dim=Dim)

        if Exper_folder is not None:
            if not os.path.exists(f'{Exper_folder}/true_f/{plot_type}/'):
                os.makedirs(f'{Exper_folder}/true_f/{plot_type}/')
            name = obj_fun.task_name
            if '.' in obj_fun.task_name:
                name = name.replace('.', '|')
            save_load = f'{Exper_folder}/true_f/{plot_type}/{name}'

        if plot_type == "1D":
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))
            test_x = np.arange(-5, 5.05, 0.005, dtype=dtype)
            test_x = test_x[:, np.newaxis]
            dic_list = []
            for i in range(len(test_x)):
                for j in range(Dim):
                    dic_list.append({f'x{j}': test_x[i][j]})
            test_y = []
            for j in range(len(test_x)):
                test_y.append(obj_fun.f(dic_list[j])['function_value'])
            test_y = np.array(test_y)
            ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
            ax.legend(['True f(x)'])
            ax.set_title(fun)
            plt.savefig(save_load)
            plt.close(fig)
        elif plot_type == "2D":
            x = np.linspace(-5, 5, 101)
            y = np.linspace(-5, 5, 101)
            X, Y = np.meshgrid(x, y)
            all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
            v_name = [f'x{i}' for i in range(Dim)]
            dic_list = [dict(zip(v_name, sample)) for sample in all_sample]
            Z_true = []
            for j in range(len(all_sample)):
                Z_true.append(obj_fun.f(dic_list[j])['function_value'])
            Z_true = np.asarray(Z_true)
            Z_true = Z_true[:, np.newaxis]
            Z_true = Z_true.reshape(X.shape)

            optimizers = obj_fun.optimizers
            fig = plt.figure(figsize=(10, 8))
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
            a = plt.contourf(X, Y, Z_true, 100, cmap=plt.cm.summer)
            plt.colorbar(a)
            plt.title(fun)
            plt.draw()
            plt.savefig(save_load, dpi=300)
            plt.close(fig)
        elif plot_type == "3D":
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            x = np.linspace(-5, 5, 101)
            y = np.linspace(-5, 5, 101)
            X, Y = np.meshgrid(x, y)
            all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
            v_name = [f'x{i}' for i in range(Dim)]
            dic_list = [dict(zip(v_name, sample)) for sample in all_sample]
            Z_true = []
            for j in range(len(all_sample)):
                Z_true.append(obj_fun.f(dic_list[j])['function_value'])
            Z_true = np.asarray(Z_true)
            Z_true = Z_true[:, np.newaxis]
            Z_true = Z_true.reshape(X.shape)
            a = ax.plot_surface(X, Y, Z_true, cmap=plt.cm.summer)
            plt.colorbar(a)
            plt.draw()
            plt.savefig(save_load, dpi=300)
            plt.close(fig)


def get_problem(fun, seed, Dim):
    # 从注册表中获取任务类
    task_class = benchmark_registry.get(fun)
    if task_class is not None:
        problem = task_class(task_name=f'{fun}_{1}',
                             task_id=1,
                             budget=100000,
                             seed=seed,
                             params={'input_dim':Dim}
                             )
    return problem

if __name__ == '__main__':
    obj_fun_list = [
        'Ellipsoid',
        'Discus',
        'BentCigar',
        'SharpRidge',
        'GriewankRosenbrock',
        'Katsuura',
    ]

    Dim = 1
    plot_true_function(obj_fun_list, Dim, np.float64, '../../experiments/plot_problem', plot_type='1D')
    Dim = 2
    plot_true_function(obj_fun_list, Dim, np.float64, '../../experiments/plot_problem', plot_type='2D')
    plot_true_function(obj_fun_list, Dim, np.float64, '../../experiments/plot_problem', plot_type='3D')