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
import matplotlib.pyplot as plt



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





def plot_testsuits(tasks):
    testsuits = ConstructLFLTestSuits(tasks)



if __name__ == '__main__':
    tasks = {
             'cp': {'budget': 8, 'time_stamp': 2, 'params':{'input_dim':2}},
             # 'MPB': {'budget': 110, 'time_stamp': 3},
             # 'Griewank': {'budget': 11, 'time_stamp': 3,  'params':{'input_dim':1}},
             # 'DixonPrice': {'budget': 110, 'time_stamp': 3},
             # 'Lunar': {'budget': 110, 'time_stamp': 3},
             # 'XGB': {'budget': 110, 'time_stamp': 3},
             }
