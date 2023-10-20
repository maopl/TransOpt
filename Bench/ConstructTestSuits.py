import copy
import numpy as np
import os
from typing import Dict
from Bench.HPO import HPOSVM, HPOXGBoost, HPORes18
from notused_py import HPOCNN, HPOMLP
from Bench.Synthetic.SyntheticBenchmark import AckleyOptBenchmark
from Bench.RL.LunarlanderBenchmark import LunarlanderBenchmark
from Bench.abstract_bench.TransferOptBenchmark import TransferOptBenchmark
from Bench.HPOB.HpobBench import HPOb
from Bench.Synthetic.MovingPeakBenchmark import MovingPeakBenchmark, MovingPeakGenerator
from Util.Register import benchmark_registry

class change_generator():
    def __init__(self, time_stamp_num, change_mode, Xdim, seed=0):
        self._time_stamp_num = time_stamp_num
        self._change_mode = change_mode
        self.num = 0

        if change_mode[0] == 'p':
            period = int(change_mode[1:])
            repeat_num = int(time_stamp_num / period) + 1
            y = np.linspace(0, 1, period) * 0.5
            y = np.tile(y, repeat_num)[:time_stamp_num]
            self.shift = np.tile(np.random.random(size=(Xdim,1)).T * 0.5,(time_stamp_num,1)) + np.tile(y[:,None], (1,Xdim))
            self.stretch = np.tile(np.random.random(size=(Xdim,1)).T * 0.2,(time_stamp_num,1)) + np.tile(y[:,None], (1,Xdim)) + 0.8
        if change_mode[0] == 's':
            self.shift = np.random.random(size=(Xdim, time_stamp_num)).T
            self.stretch = np.random.random(size=(Xdim, time_stamp_num)).T * 0.4 + 0.8

    def __iter__(self):
        return self


    def __next__(self):
        if  self.num >= self._time_stamp_num:
            raise StopIteration
        shift = self.shift[self.num]
        stretch = self.stretch[self.num]
        self.num += 1
        return shift, stretch


def get_testsuits(tasks, args):
    test_suits = ConstructLFLTestSuits(tasks=tasks, seed=args.seed)
    return test_suits


def ConstructLFLTestSuits(tasks:Dict = None, seed=0):
    test_suits = TransferOptBenchmark(seed)
    if tasks is not None:
        for task_name, task_params in tasks.items():
            fun = task_name
            budget = task_params['budget']
            time_stamp_num = task_params['time_stamp']
            params = task_params['params']

            # 从注册表中获取任务类
            task_class = benchmark_registry.get(fun)

            if task_class is not None:
                for t in range(time_stamp_num):
                    # 使用任务类构造任务对象
                    problem = task_class(task_name=f'{fun}_{t}',
                                         task_id=t,
                                         budget=budget,
                                         seed=seed,
                                         params = params,
                                         )
                    test_suits.add_task(problem)
            else:
                # 处理任务名称不在注册表中的情况
                print(f"Task '{fun}' not found in the task registry.")
                raise NameError

        return test_suits
    #
    # if tasks is not None:
    #     for f_id, function_name in enumerate(tasks):
    #         fun = function_name.split('_')[0]
    #         time_stamp_num = int(function_name.split('_')[1])
    #         if fun in ['XGB', 'SVM', 'RES']:
    #             for t in range(time_stamp_num):
    #                 if fun == 'XGB':
    #
    #                     problem = HPOXGBoost.XGBoostBenchmark(task_name=f'{fun}_{t}',
    #                                                           task_id=t,
    #                                                           budget=budget_list[f_id],seed=seed)
    #
    #                     test_suits.add_task(problem)
    #                 elif fun == 'SVM':
    #                     task_lists = [167149, 167152, 167183, 126025, 126029, 167161, 167169,
    #                                   167178, 167176, 167177]
    #                     problem = HPOSVM.SupportVectorMachine(task_name=f'{fun}_{t}',
    #                                                           task_id=task_lists[t],
    #                                                           budget=budget_list[f_id], seed=seed)
    #
    #                     test_suits.add_task(problem)
    #
    #                 elif fun == 'RES':
    #                     dataset_name = ['svhn', 'cifar10', 'cifar100']
    #                     problem = HPORes18.HPOResNet(task_name=f'RES_{t}',
    #                                                  task_id=dataset_name[t],
    #                                                  budget=budget_list[f_id],
    #                                                  seed=seed)
    #
    #                     test_suits.add_task(problem)
    #
    #         elif fun == 'lunar':
    #             lunar_seeds = [2, 3, 4, 5, 10, 14, 15, 19]
    #             for t in range(time_stamp_num):
    #                 LRLD = LunarlanderBenchmark(task_name=f'Lunarlander_{t}',
    #                                             budget=budget_list[f_id],
    #                                             seed=lunar_seeds[t])
    #                 test_suits.add_task(LRLD)
    #
    #         elif fun[:3] == 'MPB':
    #             dim = int(function_name.split('_')[2])
    #             peak_num = int(fun[3])
    #             MPB = MovingPeakGenerator(n_var=dim, n_peak=peak_num,
    #                                       shift_length=3.0, height_severity=7.0, width_severity=1.0,
    #                                       lam=0.5, n_step=time_stamp_num)
    #             peaks, widths, heights = MPB.get_MPB()
    #             for t in range(time_stamp_num):
    #                 test_suits.add_task(MovingPeakBenchmark(task_name=f'MP_{t}',
    #                                                         input_dim=dim, peak=peaks[t], width=widths[t], height=heights[t],
    #                                                         budget=budget_list[f_id], seed=seed))
    #
    #         else:
    #             dim = int(function_name.split('_')[2])
    #             if not os.path.exists('{}'.format(permutation_path)):
    #                 os.makedirs('{}'.format(permutation_path))
    #             try:
    #                 shift_array = np.loadtxt(
    #                     permutation_path + function_name + f'{dim}d_seed{seed}_shift.txt')
    #                 if len(shift_array.shape) == 1:
    #                     shift_array = shift_array[:, np.newaxis]
    #                 stretch_array = np.loadtxt(
    #                     permutation_path + function_name + f'{dim}d_seed{seed}_stretch.txt')
    #                 if len(stretch_array.shape) == 1:
    #                     stretch_array = shift_array[:, np.newaxis]
    #             except:
    #                 shift_array = np.random.random(size=(dim, time_stamp_num)).T
    #                 stretch_array = np.random.random(size=(dim, time_stamp_num)).T * 0.4 + 0.8
    #                 np.savetxt(permutation_path + function_name + f'{dim}d_seed{seed}_shift.txt',
    #                            shift_array)
    #                 np.savetxt(permutation_path + function_name + f'{dim}d_seed{seed}_stretch.txt',
    #                            stretch_array)
    #             for t in range(time_stamp_num):
    #                 shift = shift_array[t]
    #                 stretch = np.ones(dim)
    #                 shift = (shift * 2 - 1) * 0.02
    #                 problem = SyntheticOptBenchmark(task_name=fun + f'_{t}',
    #                                                 input_dim=dim,
    #                                                 seed=seed, shift=shift,
    #                                                 budget=budget_list[f_id],
    #                                                 stretch=stretch)
    #                 test_suits.add_task(problem)
    # return test_suits
    #
