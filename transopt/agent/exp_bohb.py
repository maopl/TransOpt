from bohb import BOHB
import bohb.configspace as cs
from transopt.benchmark.HPOOOD.hpoood import ERMOOD, IRMOOD, MixupOOD, DANNOOD


def objective(step, alpha, beta):
    return 1 / (alpha * step + 0.1) + beta


def evaluate(params, n_iterations):
    loss = 0.0
    for i in range(int(n_iterations)):
        loss += objective(**params, step=i)
    return loss/n_iterations


class formal_obj:
    def __init__(self, f):
        self.f = f
    def new_f(self, configuration, seed: int = 0):
        results = self.f(configuration)
        return results['function_value']


configspace = cs.ConfigurationSpace({"lr": (-8, 0), "weight_decay": (-10, -5)})



NN_name = ['ERMOOD', 'IRMOOD', 'MixupOOD', 'DANNOOD']
workloads = [0, 1, 2, 3]

for nn_name in NN_name:
    for w in workloads:
        if nn_name == 'ERMOOD':
            p  = ERMOOD(task_name='', budget_type='FEs', budget=100, seed = 0, workload = w)
        elif nn_name == 'IRMOOD':
            p  = IRMOOD(task_name='', budget_type='FEs', budget=100, seed = 0, workload = w)
        elif nn_name == 'MixupOOD':
            p  = MixupOOD(task_name='', budget_type='FEs', budget=100, seed = 0, workload = w)
        elif nn_name == 'DANNOOD':
            p  = DANNOOD(task_name='', budget_type='FEs', budget=100, seed = 0, workload = w)
        else:
            raise ValueError('Unknown task %s' % nn_name)
            
        opt = BOHB(configspace, evaluate, max_budget=100, min_budget=1)
        logs = opt.optimize()
        # for i in range(100):
        #     rec = opt.suggest(n_suggestions = 1)
        #     opt.observe(rec, p.f(configuration = rec))
        #     print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))





    # Parallel
    # opt = BOHB(configspace, evaluate, max_budget=10, min_budget=1, n_proc=4)

    
    print(logs)