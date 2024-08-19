import pandas as pd
import numpy  as np
from external.hebo.design_space.design_space import DesignSpace
from external.hebo.optimizers.hebo import HEBO

from transopt.benchmark.HPOOOD.hpoood import ERMOOD, IRMOOD, MixupOOD, DANNOOD

from transopt.agent.registry import model_registry
        


NN_name = ['ERMOOD', 'IRMOOD', 'MixupOOD', 'DANNOOD']
workloads = [0, 1, 2, 3]
seed = 0
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
            
        space = DesignSpace().parse([{'name' : 'lr', 'type' : 'num', 'lb' : -8, 'ub' : 0}, 
                                     {'name' : 'weight_decay', 'type' : 'num', 'lb' : -10, 'ub' : -5}])
        opt   = HEBO(space)
        for i in range(100):
            rec = opt.suggest(n_suggestions = 1)
            f_val = p.f(configuration = rec.to_dict(orient='records')[0])['function_value']
            print(f_val)
            y = np.array([[f_val]])
            opt.observe(rec, y)
            print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))
        

        

    
    