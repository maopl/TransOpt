import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from transopt.agent.registry import acf_registry
from transopt.optimizer.acquisition_function.ACF import AcquisitionBase


@acf_registry.register('DE-Generation')
class DEGeneration(AcquisitionBase):
    analytical_gradient_prediction = False

    def __init__(self, config):
        super(DEGeneration, self).__init__()
        if 'k' in config:
            self.k = config['k']
        else:
            self.k = 1
        if 'pop_size' in config:
            self.pop_size = config['pop_size']
        self.model = None
        self.problem = None

    def link(self, model, space):
        if self.model is None:
            self.link_model(model=model)
            self.link_space(space=space)
            self.problem = EAProblem(space.config_space, self.model.predict)
            self.ea = DE(self.pop_size)
            self.ea.setup(self.problem, verbose=False)
        else:
            self.link_model(model=model)
            self.link_space(space=space)
            self.problem = EAProblem(space.config_space, self.model.predict)

    def optimize(self):
        for i in range(self.k):
            pop = self.ea.ask()
        self.ea.evaluator.eval(self.problem, pop)
        pop_X = np.array([p.X for p in pop])
        pop_F = np.array([p.F for p in pop])
        top_k_idx = range(len(pop_X))
        elites = pop_X
        return elites

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