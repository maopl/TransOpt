import numpy as np
from pymoo.core.problem import Problem
from GPyOpt import Design_space
from pymoo.algorithms.soo.nonconvex.pso import PSO
from transopt.agent.registry import acf_registry
from transopt.optimizer.acquisition_function.ACF import AcquisitionBase


@acf_registry.register('PSO-Best')
class PSOBest(AcquisitionBase):
    analytical_gradient_prediction = False

    def __init__(self, config):
        super(PSOBest, self).__init__()
        if 'k' in config:
            self.k = config['k']
        else:
            self.k = 1
        if 'pop_size' in config:
            self.pop_size = config['pop_size']
        else:
            self.pop_size = 10
        self.model = None
        self.ea = None
        self.problem = None

    def link_space(self, space):
        opt_space = []
        for var_name in space.variables_order:
            var_dic = {
                'name': var_name,
                'type': 'continuous',
                'domain': space[var_name].search_space_range,
            }
            if space[var_name].type == 'categorical' or 'integer':
                var_dic['type'] = 'discrete'

            opt_space.append(var_dic.copy())
            
        self.space = Design_space(opt_space)

        if self.ea is None:
            self.problem = EAProblem(self.space.config_space, self.model.predict)
            self.ea = PSO(self.pop_size)
            self.ea.setup(self.problem, verbose=False)
        else:
            self.problem = EAProblem(self.space.config_space, self.model.predict)

    def optimize(self, duplicate_manager=None):
        pop = self.ea.ask()
        self.ea.evaluator.eval(self.problem, pop)
        pop_X = np.array([p.X for p in pop])
        pop_F = np.array([p.F for p in pop])
        top_k_idx = sorted(range(len(pop_F)), key=lambda i: pop_F[i])[:self.k]
        elites = pop_X[top_k_idx]
        elites_F = pop_F[top_k_idx]
        return elites, elites_F

    def _compute_acq(self, x):
        raise NotImplementedError()

    def _compute_acq_withGradients(self, x):
        raise NotImplementedError()


class EAProblem(Problem):
    def __init__(self, space, predict):
        input_dim = len(space)
        xl = []
        xu = []
        for var_info in space:
            var_domain = var_info['domain']
            xl.append(var_domain[0])
            xu.append(var_domain[1])
        xl = np.array(xl)
        xu = np.array(xu)
        self.predict = predict
        super().__init__(n_var=input_dim, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"], _ = self.predict(x)