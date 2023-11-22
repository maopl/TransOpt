import GPy
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from typing import Dict, Union, List
from Optimizer.BayesianOptimizerBase import BayesianOptimizerBase
from Util.Data import ndarray_to_vectors
from Util.Register import optimizer_register
from Util.Normalization import get_normalizer


@optimizer_register('SAEA')
class SurrogateAssistedEA(BayesianOptimizerBase):
    def __init__(self, config: Dict, **kwargs):
        super(SurrogateAssistedEA, self).__init__(config=config)

        self.init_method = 'latin'
        self.model = None
        self.ea = None
        self.problem = None

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = True

        if 'init_number' in config:
            self.ini_num = config['init_number']
        else:
            self.ini_num = None

        if 'surrogate_mode' in config:
            self.surrogate_mode = config['surrogate_mode']
        else:
            self.surrogate_mode = 'Kriging'

        if 'ea' in config:
            self.ea_name = config['ea']
        else:
            self.ea_name = 'GA'

    def initial_sample(self):
        return self.sample(self.ini_num)

    def suggest(self, n_suggestions: Union[None, int] = None) -> List[Dict]:
        if self._X.size == 0:
            suggests = self.initial_sample()
            return suggests
        else:
            if 'normalize' in self.config:
                self.normalizer = get_normalizer(self.config['normalize'])

            Data = {'Target': {'X': self._X, 'Y': self._Y}}
            self.update_model(Data)
            self.problem = EAProblem(self.search_space.config_space, self.predict)
            # 得到新的种群
            pop = self.ea.ask()
            self.ea.evaluator.eval(self.problem, pop)
            pop_X = np.array([p.X for p in pop])
            pop_F = np.array([p.F for p in pop])
            # 选择需要准确评估的个体
            elites_idx = np.argsort(pop_F)[0]
            elites = pop_X[elites_idx]
            # 准确评估优秀个体
            suggested_sample = self.search_space.zip_inputs(elites)
            suggested_sample = ndarray_to_vectors(self._get_var_name('search'), suggested_sample)
            design_suggested_sample = self.inverse_transform(suggested_sample)
            elites_obj = self.testsuits.f(design_suggested_sample)
            pop[elites_idx].F = elites_obj
            # 将 pop 返回给 EA
            self.ea.tell(infills=pop)

            return design_suggested_sample


    def update_model(self, Data):
        assert 'Target' in Data
        target_data = Data['Target']
        X = target_data['X']
        Y = target_data['Y']

        if self.normalizer is not None:
            Y = self.normalizer(Y)

        if self.obj_model == None:
            self.create_model(X, Y)
            self.problem = EAProblem(self.search_space.config_space, self.predict)
            self.create_ea()
        else:
            self.obj_model.set_XY(X, Y)

        try:
            self.obj_model.optimize_restarts(num_restarts=1, verbose=self.verbose, robust=True)
        except np.linalg.LinAlgError as e:
            # break
            print('Error: np.linalg.LinAlgError')

    def create_model(self, X, Y):
        kern = GPy.kern.RBF(self.input_dim, ARD=True)
        self.obj_model = GPy.models.GPRegression(X, Y, kernel=kern)
        self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)

    def create_ea(self):
        self.ea = GA(self.ini_num)
        self.ea.setup(self.problem, verbose=False)

    def predict(self, X):
        if X.ndim == 1:
            X = X[None, :]

        m, v = self.obj_model.predict(X)
        return m, v

    def sample(self, num_samples: int) -> List[Dict]:
        if self.input_dim is None:
            raise ValueError("Input dimension is not set. Call set_search_space() to set the input dimension.")

        temp = None
        if self.init_method == 'latin':
            temp = np.random.rand(num_samples, self.input_dim)
            for i in range(self.input_dim):
                temp[:, i] = (temp[:, i] + np.random.permutation(np.arange(num_samples))) / num_samples

        samples = []
        for i in range(num_samples):
            sample = {}
            for j, var_info in enumerate(self.search_space.config_space):
                var_name = var_info['name']
                var_domain = var_info['domain']
                if self.init_method == 'random':
                    value = np.random.uniform(var_domain[0], var_domain[1])
                elif self.init_method == 'latin':
                    value = temp[i][j] * (var_domain[1] - var_domain[0]) + var_domain[0]
                sample[var_name] = value
            samples.append(sample)

        samples = self.inverse_transform(samples)
        return samples

    def model_reset(self):
        self.obj_model = None

    def get_fmin(self):
        m, v = self.predict(self.obj_model.X)
        return m.min()

    def optimize(self, testsuits, data_handler):
        self.set_DataHandler(data_handler)
        while (testsuits.get_unsolved_num()):
            space_info = testsuits.get_cur_space_info()
            self.reset(testsuits.get_curname(), space_info, search_sapce=None)
            testsuits.sync_query_num(len(self._X))
            self.set_auxillary_data()
            while (testsuits.get_rest_budget()):
                self.testsuits = testsuits
                suggested_sample = self.suggest()
                # testsuits.lock()
                observation = testsuits.f(suggested_sample)
                # testsuits.unlock()
                self.observe(suggested_sample, observation)
            testsuits.roll()

    def reset(self, task_name:str, design_space:Dict, search_sapce:Union[None, Dict] = None):
        self.set_space(design_space, search_sapce)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self._data_handler.reset_task(task_name, design_space)
        self.sync_data(self._data_handler.get_input_vectors(), self._data_handler.get_output_value())
        self.model_reset()


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
