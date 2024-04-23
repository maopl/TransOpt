import GPy
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from typing import Dict, Union, List

from transopt.optimizer.optimizer_base import BOBase
from transopt.utils.serialization import vectors_to_ndarray, output_to_ndarray
from transopt.utils.serialization import ndarray_to_vectors
from agent.registry import optimizer_register
from transopt.utils.Normalization import get_normalizer


@optimizer_register('KrigingGA')
class KrigingGA(BOBase):
    def __init__(self, config: Dict, **kwargs):
        super(KrigingGA, self).__init__(config=config)

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

        self.pop = None

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
            self.pop = self.ea.ask()
            self.ea.evaluator.eval(self.problem, self.pop)
            pop_X = np.array([p.X for p in self.pop])
            pop_F = np.array([p.F for p in self.pop])
            # 选择需要准确评估的个体
            self.elites_idx = np.argsort(pop_F)[0]
            elites = pop_X[self.elites_idx]
            # 准确评估优秀个体
            suggested_sample = self.search_space.zip_inputs(elites)
            suggested_sample = ndarray_to_vectors(self._get_var_name('search'), suggested_sample)
            design_suggested_sample = self.inverse_transform(suggested_sample)

            return design_suggested_sample

    def observe(self, input_vectors: Union[List[Dict], Dict], output_value: Union[List[Dict], Dict]) -> None:
        self._data_handler.add_observation(input_vectors, output_value)

        # Convert dict to list of dict
        if isinstance(input_vectors, Dict):
            input_vectors = [input_vectors]
        if isinstance(output_value, Dict):
            output_value = [output_value]

        # Check if the lists are empty and return if they are
        if len(input_vectors) == 0 and len(output_value) == 0:
            return


        self._validate_observation('design', input_vectors=input_vectors, output_value=output_value)
        X = self.transform(input_vectors)

        self._X = np.vstack((self._X, vectors_to_ndarray(self._get_var_name('search'), X))) if self._X.size else vectors_to_ndarray(self._get_var_name('search'), X)
        self._Y = np.vstack((self._Y, output_to_ndarray(output_value))) if self._Y.size else output_to_ndarray(output_value)

        if self.pop is not None:
            self.pop[self.elites_idx].F = input_vectors
            # 将 pop 返回给 EA
            self.ea.tell(infills=self.pop)

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