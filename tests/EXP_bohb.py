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
    def new_f(self, params, n_iterations):
        # lr = params['lr']
        # weight_decay = params['weight_decay']
        results = self.f(params, epoch = n_iterations)
        return results['function_value']


lr = cs.UniformHyperparameter('lr', upper=0, lower=-8)
weight_decay = cs.UniformHyperparameter('weight_decay', upper=-5, lower=-10)
configspace = cs.ConfigurationSpace([lr, weight_decay])


NN_name = ['MixupOOD', 'DANNOOD']
workloads = [0, 1, 2, 3]
seed = 1
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
        
        new_obj = formal_obj(p.f)
            
        opt = BOHB(configspace, new_obj.new_f, max_budget=100, min_budget=1)
        logs = opt.optimize()
        # for i in range(100):
        #     rec = opt.suggest(n_suggestions = 1)
        #     opt.observe(rec, p.f(configuration = rec))
        #     print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))





    # Parallel
    # opt = BOHB(configspace, evaluate, max_budget=10, min_budget=1, n_proc=4)

    
    print(logs)