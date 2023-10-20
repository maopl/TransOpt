
# %matplotlib notebook

import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, Dict
import ConfigSpace as CS
from Util.Register import benchmark_register, benchmark_registry
from Bench.abstract_bench.ConfigOptBenchmark import ContinuousOptBenchmark
logger = logging.getLogger('SyntheticBenchmark')


@benchmark_register('Sphere')
class SphereOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        super(SphereOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.sum((X) ** 2, axis=1)
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.12, upper=5.12) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}




@benchmark_register('Rastrigin')
class RastriginOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift + 2.0)
        self.dtype =np.float64

        super(RastriginOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.4)

        n = X.shape[0]
        d = X.shape[1]

        pi = np.array([math.pi], dtype=self.dtype)
        y = 10.0 * self.input_dim + np.sum((X) ** 2 - 10.0 * np.cos(pi * (X)), axis=1)
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.12, upper=5.12) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}



@benchmark_register('Schwefel')
class SchwefelOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(420.9687 - self.shift)
        self.dtype =np.float64

        super(SchwefelOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = 420 - np.sum(np.multiply(X, np.sin(np.sqrt(abs(self.stretch * X - self.shift)))), axis=1)
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-500.0, upper=500.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}



@benchmark_register('LevyR')
class LevyROptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift - 1.0)
        self.dtype =np.float64

        super(LevyROptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.1)

        n = X.shape[0]
        d = X.shape[1]

        w = 1.0 + X / 4.0
        pi = np.array([math.pi], dtype=self.dtype)
        part1 = np.sin(pi * w[..., 0]) ** 2
        part2 = np.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 5.0 * np.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            axis=1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
                1.0 + np.sin(2 * math.pi * w[..., -1]) ** 2
        )
        y = part1 + part2 + part3
        # y +=  self.noise(n)

        return {'function_value': float(-y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-10.0, upper=10.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('Griewank')
class GriewankOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        super(GriewankOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        div = np.arange(start=1, stop=d + 1, dtype=self.dtype)
        part1 = np.sum(X ** 2 / 4000.0, axis=1)
        part2 = -np.prod(np.cos(X / np.sqrt(div)), axis=1)
        y = part1 + part2 + 1.0
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-100.0, upper=100.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('Rosenbrock')
class RosenbrockOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        super(RosenbrockOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            axis=-1,
        )
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.0, upper=10.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}

@benchmark_register('DropwaveR')
class DropwaveROptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift + 3.3)
        self.dtype =np.float64

        self.a = np.array([20], dtype=self.dtype)
        self.b = np.array([0.2], dtype=self.dtype)
        self.c = np.array([2 * math.pi], dtype=self.dtype)

        super(DropwaveROptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.33)

        n = X.shape[0]
        d = X.shape[1]

        part1 = np.linalg.norm(X, axis=1)
        y = -(3 + np.cos(part1)) / (0.1 * np.power(part1, 1.5) + 1)
        # y +=  self.noise(n)

        return {'function_value': float(-y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-10.0, upper=10.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('Langermann')
class LangermannOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        self.c = np.array([1, 2, 5])
        self.m = 3
        self.A = np.random.randint(1, 10, (self.m, self.input_dim))

        super(LangermannOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = 0
        for i in range(self.m):
            part1 = np.exp(-np.sum(np.power(X - self.A[i], 2), axis=1) / np.pi)
            part2 = np.cos(np.sum(np.power(X - self.A[i], 2), axis=1) * np.pi)
            y += part1 * part2 * self.c[i]
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=0, upper=10.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}



@benchmark_register('RotatedHyperEllipsoid')
class RotatedHyperEllipsoidOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift - 32.75)
        self.dtype =np.float64


        super(RotatedHyperEllipsoidOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift + 0.5)

        n = X.shape[0]
        d = X.shape[1]

        div = np.arange(start=d, stop=0, step=-1, dtype=self.dtype)
        y = np.sum(div * X ** 2, axis=1)
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-65.536, upper=65.536) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('SumOfDifferentPowers')
class SumOfDifferentPowersOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift + 0.238)
        self.dtype =np.float64


        super(SumOfDifferentPowersOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.238)

        n = X.shape[0]
        d = X.shape[1]

        y = np.zeros(shape=(n,), dtype=self.dtype)
        for i in range(d):
            y += np.abs(X[:, i]) ** (i + 1)
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-1.0, upper=1.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}




@benchmark_register('StyblinskiTang')
class StyblinskiTangOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift - 2.903534)
        self.dtype =np.float64


        super(StyblinskiTangOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(axis=1)
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.0, upper=5.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('Powell')
class PowellOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = [tuple(0.0 for _ in range(self.input_dim))]
        self.dtype =np.float64


        super(PowellOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.zeros_like(X[..., 0])
        for i in range(self.input_dim // 4):
            i_ = i + 1
            part1 = (X[..., 4 * i_ - 4] + 10.0 * X[..., 4 * i_ - 3]) ** 2
            part2 = 5.0 * (X[..., 4 * i_ - 2] - X[..., 4 * i_ - 1]) ** 2
            part3 = (X[..., 4 * i_ - 3] - 2.0 * X[..., 4 * i_ - 2]) ** 4
            part4 = 10.0 * (X[..., 4 * i_ - 4] - X[..., 4 * i_ - 1]) ** 4
            y += part1 + part2 + part3 + part4
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-4.0, upper=5.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('DixonPrice')
class DixonPriceOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = [
            tuple(
                math.pow(2.0, -(1.0 - 2.0 ** (-(i - 1))))
                for i in range(1, self.input_dim + 1)
            )
        ]
        self.dtype =np.float64


        super(DixonPriceOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        part1 = (X[..., 0] - 1) ** 2
        i = np.arange(start=2, stop=d + 1, step=1)
        i = np.tile(i, (n, 1))
        part2 = np.sum(i * (2.0 * X[..., 1:] ** 2 - X[..., :-1]) ** 2, axis=1)
        y = part1 + part2
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-10.0, upper=10.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}



@benchmark_register('cp')
class cpOptBenchmark(ContinuousOptBenchmark):
    def __init__(self, task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = [
            tuple(
                math.pow(2.0, -(1.0 - 2.0 ** (-(i - 1))))
                for i in range(1, self.input_dim + 1)
            )
        ]
        self.dtype =np.float64


        super(cpOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        n = X.shape[0]
        d = X.shape[1]

        part1 = np.sin(6 * X)
        # part2 =  0.2 * X**2
        # if self.shift == -1:
        #     part3 = 0.1 * ((3) * (X + 0.5)) ** 2
        # else:
        #     part3 = 0.1 * ((3) * (X - 0.5)) ** 2
        part3 = 0.1 * ((3) * (X - 0.5)) ** 2
        y = part1 + part3
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-1, upper=1) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}




@benchmark_register('Ackley')
class AckleyOptBenchmark(ContinuousOptBenchmark):
    def __init__(self, task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift - 12)
        self.dtype =np.float64

        self.a = np.array([20], dtype=self.dtype)
        self.b = np.array([0.2], dtype=self.dtype)
        self.c = np.array([0.3 * math.pi], dtype=self.dtype)


        super(AckleyOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.73)

        n = X.shape[0]
        d = X.shape[1]
        a, b, c = self.a, self.b, self.c

        part1 = -a * np.exp(-b / math.sqrt(d) * np.linalg.norm(X, axis=-1))
        part2 = -(np.exp(np.mean(np.cos(c * X), axis=-1)))
        y = part1 + part2 + a + math.e
        # y +=  self.noise(n)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-32.768, upper=32.768) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space


    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('Ellipsoid')
class EllipsoidOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        self.condition = 1e6

        super(EllipsoidOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            temp = x[0] * x[0]
            for i in range(1, d):
                exponent = 1.0 * i / (d - 1)
                temp += pow(self.condition, exponent) * x[i] * x[i]
            y = np.append(y, temp)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.0, upper=5.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('Discus')
class DiscusOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        self.condition = 1e6

        super(DiscusOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            temp = self.condition * x[0] * x[0]
            for i in range(1, d):
                temp += x[i] * x[i]
            y = np.append(y, temp)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.0, upper=5.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('BentCigar')
class BentCigarOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        self.condition = 1e6

        super(BentCigarOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            temp = x[0] * x[0]
            for i in range(1, d):
                temp += self.condition * x[i] * x[i]
            y = np.append(y, temp)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.0, upper=5.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('SharpRidge')
class SharpRidgeOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        self.alpha = 100.0

        super(SharpRidgeOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        d_vars_40 = d / 40.0
        vars_40 = int(math.ceil(d_vars_40))
        y = np.array([])
        for x in X:
            temp = 0
            for i in range(vars_40, d):
                temp += x[i] * x[i]
            temp = self.alpha * math.sqrt(temp / d_vars_40)
            for i in range(vars_40):
                temp += x[i] * x[i] / d_vars_40
            y = np.append(y, temp)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.0, upper=5.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        print(1)
        return {}



@benchmark_register('GriewankRosenbrock')
class GriewankRosenbrockOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        super(GriewankRosenbrockOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            temp = 0
            for i in range(len(x) - 1):
                temp1 = x[i] * x[i] - x[i+1]
                temp2 = 1.0 - x[i]
                temp3 = 100.0 * temp1 ** 2 + temp2 ** 2
                temp += temp3 / 4000.0 - math.cos(temp3)
            y = np.append(y, temp)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.0, upper=5.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        print(1)
        return {}


@benchmark_register('Katsuura')
class KatsuuraOptBenchmark(ContinuousOptBenchmark):
    def __init__(self,task_name, budget, seed, task_id, task_type='non-tabular', **kwargs):
        assert 'params' in kwargs
        parameters = kwargs['params']
        self.input_dim = parameters['input_dim']

        if 'shift' in parameters:
            self.shift = parameters['shift']
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if 'stretch' in parameters:
            self.stretch = parameters['stretch']
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype =np.float64

        super(KatsuuraOptBenchmark, self).__init__(task_name=task_name, seed=seed, task_id=task_id, task_type=task_type, budget=budget)

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            result = 1.0
            for i in range(len(x)):
                temp = 0.0
                for j in range(1, 33):
                    temp1 = 2.0 ** j
                    temp += abs(temp1 * x[i] - round(temp1 * x[i])) / temp1
                temp = 1.0 + (i + 1) * temp
                result *= temp ** (10.0 / (len(x) ** 1.2))
            y = np.append(y, result)

        return {'function_value': float(y),
                'info': {'fidelity': fidelity}
                }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'x{i}', lower=-5.0, upper=5.0) for i in range(self.input_dim)])

        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        print(1)
        return {}

# def plot_true_oned(obj_fun_list, Dim, dtype, Exper_floder=None):
#     for i in obj_fun_list:
#         f, ax = plt.subplots(1, 1, figsize=(16, 6))
#         problem = get_problem(i, seed=0, Dim=Dim)
#         # bounds = problem.bounds
#         opt_x = problem.optimizers
#         # opt_val = problem.optimal_value
#         test_x = np.arange(-5, 5.05, 0.005, dtype=dtype)
#         test_x = test_x[:, np.newaxis]
#         dic_list = []
#         for i in range(len(test_x)):
#             for j in range(Dim):
#                 dic_list.append({f'x{j}':test_x[i][j]})
#         test_y = []
#         for j in range(len(test_x)):
#             test_y.append(problem.f(dic_list[j])['function_value'])
#         test_y = np.array(test_y)
#         # test_y = Normalize(test_y)
#         ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
#         ax.legend(['True f(x)'])
#         # ax.set_xlim([bounds[0][0], bounds[1][0]])
#         ax.set_title(i)
#         # plt.show()
#         if not os.path.exists('{}/true_f/oneD/'.format(Exper_floder)):
#             os.makedirs('{}/true_f/oneD/'.format(Exper_floder))
#         name = problem.task_name
#         if '.' in problem.task_name:
#             name = name.replace('.','|')
#
#         save_load = '{}/true_f/oneD/{}'.format(Exper_floder, name)
#
#         plt.savefig(save_load+'')
#
# def plot_true_contour(obj_fun_list, Dim, dtype, Exper_floder=None):
#     for fun in obj_fun_list:
#         obj_fun = get_problem(fun, seed=0, Dim=Dim)
#
#         if not os.path.exists('{}/true_f/contour/'.format(Exper_floder, obj_fun.task_name)):
#             os.makedirs('{}/true_f/contour/'.format(Exper_floder, obj_fun.task_name))
#         name = obj_fun.task_name
#         if '.' in obj_fun.task_name:
#             name = name.replace('.','|')
#         save_load = '{}/true_f/contour/{}'.format(Exper_floder, name)
#
#         x = np.linspace(-5, 5, 101)
#         y = np.linspace(-5, 5, 101)
#         X, Y = np.meshgrid(x, y)
#         all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
#         v_name = ["x", "y"]
#         dic_list = [dict(zip(v_name, sample)) for sample in all_sample]
#         Z_true = []
#         for j in range(len(all_sample)):
#             Z_true.append(obj_fun.f(dic_list[j])['function_value'])
#         Z_true = np.asarray(Z_true)
#         Z_true = Z_true[:,np.newaxis]
#         Z_true = Z_true.reshape(X.shape)
#
#         optimizers = obj_fun.optimizers
#
#         fig = plt.figure(figsize=(10, 8))
#         ax = plt.subplot(111)
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
#         a = plt.contourf(X, Y, Z_true, 100, cmap=plt.cm.summer)
#         # b = plt.contour(X, Y, Z_true, 50, colors='black', linewidths=1, linestyles='solid')
#         # plt.plot(optimizers[:, 0], optimizers[:, 1], marker='*', linewidth=0, color='white', markersize=10, label="GlobalOpt")
#         plt.colorbar(a)
#         plt.title(fun)
#         # fig.legend(facecolor='gray')
#         plt.draw()
#         plt.savefig(save_load, dpi=300)
#         plt.close()
#
#
# def plot_true_3D(obj_fun_list, Dim, dtype, Exper_floder=None):
#     for fun in obj_fun_list:
#         obj_fun = get_problem(fun, seed=0, Dim=Dim)
#
#         fig = plt.figure()  # 定义新的三维坐标轴
#         ax = plt.axes(projection='3d')
#
#         # 定义三维数据
#         x = np.linspace(-5, 5, 101)
#         y = np.linspace(-5, 5, 101)
#         X, Y = np.meshgrid(x, y)
#         all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
#         v_name = ["x", "y"]
#         dic_list = [dict(zip(v_name, sample)) for sample in all_sample]
#         Z_true = []
#         for j in range(len(all_sample)):
#             Z_true.append(obj_fun.f(dic_list[j])['function_value'])
#         Z_true = np.asarray(Z_true)
#         Z_true = Z_true[:, np.newaxis]
#         Z_true = Z_true.reshape(X.shape)
#
#
#         # 作图
#         a = ax.plot_surface(X, Y, Z_true, cmap=plt.cm.summer)
#         if not os.path.exists('{}/true_f/3D/'.format(Exper_floder, obj_fun.task_name)):
#             os.makedirs('{}/true_f/3D/'.format(Exper_floder, obj_fun.task_name))
#         name = obj_fun.task_name
#         if '.' in obj_fun.task_name:
#             name = name.replace('.','|')
#         save_load = '{}/true_f/3D/{}'.format(Exper_floder, name)
#         plt.colorbar(a)
#         plt.draw()
#         # plt.show()
#         # print(1)
#         plt.savefig(save_load, dpi=300)


def plot_true_function(obj_fun_list, Dim, dtype, Exper_folder=None, plot_type="1D"):
    for fun in obj_fun_list:
        obj_fun = get_problem(fun, seed=0, Dim=Dim)

        if Exper_folder is not None:
            if not os.path.exists(f'{Exper_folder}/true_f/{plot_type}/'):
                os.makedirs(f'{Exper_folder}/true_f/{plot_type}/')
            name = obj_fun.task_name
            if '.' in obj_fun.task_name:
                name = name.replace('.', '|')
            save_load = f'{Exper_folder}/true_f/{plot_type}/{name}'

        if plot_type == "1D":
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))
            test_x = np.arange(-5, 5.05, 0.005, dtype=dtype)
            test_x = test_x[:, np.newaxis]
            dic_list = []
            for i in range(len(test_x)):
                for j in range(Dim):
                    dic_list.append({f'x{j}': test_x[i][j]})
            test_y = []
            for j in range(len(test_x)):
                test_y.append(obj_fun.f(dic_list[j])['function_value'])
            test_y = np.array(test_y)
            ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
            ax.legend(['True f(x)'])
            ax.set_title(fun)
            plt.savefig(save_load)
            plt.close(fig)
        elif plot_type == "2D":
            x = np.linspace(-5, 5, 101)
            y = np.linspace(-5, 5, 101)
            X, Y = np.meshgrid(x, y)
            all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
            v_name = [f'x{i}' for i in range(Dim)]
            dic_list = [dict(zip(v_name, sample)) for sample in all_sample]
            Z_true = []
            for j in range(len(all_sample)):
                Z_true.append(obj_fun.f(dic_list[j])['function_value'])
            Z_true = np.asarray(Z_true)
            Z_true = Z_true[:, np.newaxis]
            Z_true = Z_true.reshape(X.shape)

            optimizers = obj_fun.optimizers
            fig = plt.figure(figsize=(10, 8))
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
            a = plt.contourf(X, Y, Z_true, 100, cmap=plt.cm.summer)
            plt.colorbar(a)
            plt.title(fun)
            plt.draw()
            plt.savefig(save_load, dpi=300)
            plt.close(fig)
        elif plot_type == "3D":
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            x = np.linspace(-5, 5, 101)
            y = np.linspace(-5, 5, 101)
            X, Y = np.meshgrid(x, y)
            all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
            v_name = [f'x{i}' for i in range(Dim)]
            dic_list = [dict(zip(v_name, sample)) for sample in all_sample]
            Z_true = []
            for j in range(len(all_sample)):
                Z_true.append(obj_fun.f(dic_list[j])['function_value'])
            Z_true = np.asarray(Z_true)
            Z_true = Z_true[:, np.newaxis]
            Z_true = Z_true.reshape(X.shape)
            a = ax.plot_surface(X, Y, Z_true, cmap=plt.cm.summer)
            plt.colorbar(a)
            plt.draw()
            plt.savefig(save_load, dpi=300)
            plt.close(fig)


def get_problem(fun, seed, Dim):
    # 从注册表中获取任务类
    task_class = benchmark_registry.get(fun)

    if task_class is not None:
        problem = task_class(task_name=f'{fun}_{1}',
                             task_id=1,
                             budget=100000,
                             seed=seed,
                             params={'input_dim':Dim}
                             )
    # if fun == "Sphere":
    #     problem = SphereOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "Rastrigin":
    #     problem = RastriginOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "Ackley":
    #     problem = AckleyOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "Schwefel":
    #     problem = SchwefelOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "LevyR":
    #     problem = LevyROptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "Griewank":
    #     problem = GriewankOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "Rosenbrock":
    #     problem = RosenbrockOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "DropwaveR":
    #     problem = DropwaveROptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "Langermann":
    #     problem = LangermannOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "RotatedHyperEllipsoid":
    #     problem = RotatedHyperEllipsoidOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "SumOfDifferentPowers":
    #     problem = SumOfDifferentPowersOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "StyblinskiTang":
    #     problem = StyblinskiTangOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "Powell":
    #     problem = PowellOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "DixonPrice":
    #     problem = DixonPriceOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    # elif fun == "cp":
    #     problem = cpOptBenchmark(task_name=fun + f'_{0}',
    #                                  budget=100000,
    #                                  seed=seed,
    #                                  task_type='non-tabular',
    #                                  params={'input_dim':Dim})
    return problem

if __name__ == '__main__':
    Dim = 2
    obj_fun_list = [
        'Sphere',
        'Rastrigin',
        'Ackley',
        'Schwefel',
        'LevyR',
        'Griewank',
        'Rosenbrock',
        'DropwaveR',
        'Langermann',
        'RotatedHyperEllipsoid',
        'SumOfDifferentPowers',
        'StyblinskiTang',
        'Powell',
        'DixonPrice',
    ]

    if Dim == 1:
        plot_true_function(obj_fun_list, Dim, np.float64, '../../experiments/plot_problem', plot_type='1D')
    elif Dim == 2:
        plot_true_function(obj_fun_list, Dim, np.float64, '../../experiments/plot_problem', plot_type='2D')
        plot_true_function(obj_fun_list, Dim, np.float64, '../../experiments/plot_problem', plot_type='3D')

    # from Bench.abstract_bench.TransferOptBenchmark import TransferOptBenchmark
    #
    # seed=0
    # test_suits = TransferOptBenchmark(seed=seed)
    #
    # for fun in obj_fun_list:
    #     shift = np.zeros(Dim)
    #     stretch = np.ones(Dim)
    #     problem = getProblem(fun, seed, Dim)
    #     test_suits.add_task(problem)
    #
    # test_x = np.arange(-5, 5.05, 0.005, dtype=np.float64)
    # test_x = test_x[:, np.newaxis]
    # dic_list = []
    #
    # for i in range(len(test_x)):
    #     for j in range(Dim):
    #         dic_list.append({f'x{j}':test_x[i][j]})
    # for i in obj_fun_list:
    #     test_y = []
    #     for j in range(len(test_x)):
    #         test_y.append(test_suits.f(dic_list[j])['function_value'])
    #     test_y = np.array(test_y)
    #     f, ax = plt.subplots(1, 1, figsize=(16, 6))
    #     ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
    #     plt.show()
    #
    #     test_suits.roll()


