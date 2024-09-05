from __future__ import annotations

import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer


from transopt.benchmark.HPOOOD.hpoood import ERMOOD, IRMOOD, MixupOOD, DANNOOD



def sphere(eval_config: dict[str, float]) -> tuple[dict[str, float], float]:
    start = time.time()
    vals = np.array(list(eval_config.values()))
    vals *= vals
    return {"loss": np.sum(vals)}, time.time() - start
    
    

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
        
        
        dim = 2
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"lr", lower=-8, upper=0))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"weight_decay", lower=-10, upper=-5))

        opt = TPEOptimizer(obj_func=p.f, config_space=cs, n_init=22, max_evals=100, resultfile='nn_name')
        # If you do not want to do logging, remove the `logger_name` argument
        print(opt.optimize())   