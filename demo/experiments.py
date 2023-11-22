import logging
import os
import argparse
import pickle
import numpy as np


import Util.Register
import KnowledgeBase.DataSelection
from Benchmark.ConstructTestSuits import get_testsuits
from Optimizer.ConstructOptimizer import get_optimizer
from KnowledgeBase.ConstructKB import get_knowledgebase
from KnowledgeBase.TaskDataHandler import OptTaskDataHandler
from KnowledgeBase.KnowledgeBaseAccessor import KnowledgeBaseAccessor


os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"


def run_experiments(tasks, args):
    logger = logging.getLogger(__name__)
    kb = get_knowledgebase(args)
    testsuits = get_testsuits(tasks, args)
    optimizer = get_optimizer(args)
    data_handler = OptTaskDataHandler(kb, args)
    optimizer.optimize(testsuits, data_handler)




if __name__ == '__main__':
    tasks = {
            # 'DBMS':{'budget': 11, 'time_stamp': 3},
            # 'GCC' : {'budget': 11, 'time_stamp': 3},
            # 'LLVM' : {'budget': 11, 'time_stamp': 3},
            #  'Rosenbrock': {'budget': 11, 'time_stamp': 3, 'params':{'input_dim':1}},
             # 'MPB': {'budget': 110, 'time_stamp': 3},
             'Sphere': {'budget': 100, 'time_stamp': 3,  'params':{'input_dim':1}},
             # 'DixonPrice': {'budget': 110, 'time_stamp': 3},
             # 'Lunar': {'budget': 110, 'time_stamp': 3},
             # 'XGB': {'budget': 110, 'time_stamp': 3},
             }

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-im", "--init_method", type=str, default='random')
    parser.add_argument("-in", "--init_number", type=int, default=2)
    parser.add_argument("-p", "--exp_path", type=str, default='../../LFL_experiments')
    parser.add_argument("-n", "--exp_name", type=str, default='test')  # 实验名称，保存在experiments中
    parser.add_argument("-s", "--seed", type=int, default=0)  # 设置随机种子，与迭代次数相关
    parser.add_argument("-m", "--optimizer", type=str, default='BO')  # 设置method:WS,MT,INC
    parser.add_argument("-norm", "--normalize", type=str, default='norm')
    parser.add_argument("-ns", "--source_num", type=int, default=2)
    parser.add_argument("-slt", "--selector", type=str, default='recent')
    parser.add_argument("-sm", "--save_mode", type=int, default=1)  # 控制是否保存模型
    parser.add_argument("-lm", "--load_mode", type=int, default=0)  # 控制是否从头开始
    parser.add_argument("-lg", "--load_gym", type=int, default=1)  #
    parser.add_argument("-mt", "--match", type=int, default=1)  #
    parser.add_argument("-ac", "--acquisition_func", type=str, default='LCB')  # 控制BO的acquisition function
    args = parser.parse_args()

    run_experiments(tasks, args)




