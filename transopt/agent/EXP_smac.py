from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
import time
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.model_selection import cross_val_score


from transopt.benchmark.HPOOOD.hpoood import ERMOOD, IRMOOD, MixupOOD, DANNOOD

class formal_obj:
    def __init__(self, f):
        self.f = f
    def new_f(self, configuration):
        start = time.time()
        results = self.f(configuration)
        return results['function_value']


configspace = ConfigurationSpace({"lr": (-8, 0), "weight_decay": (-10, -5)})

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=100)

NN_name = ['ERMOOD', 'IRMOOD', 'MixupOOD', 'DANNOOD']
workloads = [0, 1, 2, 3]
seed=0
for nn_name in NN_name:
    for w in workloads:
        if nn_name == 'ERMOOD':
            p  = ERMOOD(task_name='', budget_type='FEs', budget=100, seed = seed, workload = w)
        elif nn_name == 'IRMOOD':
            p  = IRMOOD(task_name='', budget_type='FEs', budget=100, seed = seed, workload = w)
        elif nn_name == 'MixupOOD':
            p  = MixupOOD(task_name='', budget_type='FEs', budget=100, seed = seed, workload = w)
        elif nn_name == 'DANNOOD':
            p  = DANNOOD(task_name='', budget_type='FEs', budget=100, seed = seed, workload = w)
        else:
            raise ValueError('Unknown task %s' % nn_name)
        train = formal_obj(p.f)
        
        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train.new_f)
        incumbent = smac.optimize()
        