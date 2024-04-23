import json
import random
import unittest
from unittest.mock import MagicMock
from benchmark.instantiate_problems import InstantiateProblems

tasks = {
    # 'DBMS':{'budget': 11, 'time_stamp': 3},
    # 'GCC' : {'budget': 11, 'time_stamp': 3},
    # 'LLVM' : {'budget': 11, 'time_stamp': 3},
    'Ackley': {'budget': 11, 'workloads': [1,2,3], 'params':{'input_dim':1}},
    # 'MPB': {'budget': 110, 'time_stamp': 3},
    # 'Griewank': {'budget': 11, 'time_stamp': 3,  'params':{'input_dim':2}},
    # "AckleySphere": {"budget": 1000, "workloads":[1,2,3], "params": {"input_dim": 2}},
    # 'Lunar': {'budget': 110, 'time_stamp': 3},
    # 'XGB': {'budget': 110, 'time_stamp': 3},
}



class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.problems = InstantiateProblems(tasks)
        
    
    def plot_problems(self):
        for p in self.problems:
            obj_fun_list = p.get_obj_fun_list()
            Dim = p.get_input_dim()
        if Dim == 1:
            plot_true_function(
                obj_fun_list,
                Dim,
                np.float64,
                "../../experiments/plot_problem",
                plot_type="1D",
            )
        elif Dim == 2:
            plot_true_function(
                obj_fun_list,
                Dim,
                np.float64,
                "../../experiments/plot_problem",
                plot_type="2D",
            )
            plot_true_function(
                obj_fun_list,
                Dim,
                np.float64,
                "../../experiments/plot_problem",
                plot_type="3D",
            )


if __name__ == '__main__':
    unittest.main()