""" Base-class of all benchmarks """

import abc
from typing import Union, Dict
import functools

import logging
import ConfigSpace
import numpy as np

from ConfigSpace.util import deactivate_inactive_hyperparameters

logger = logging.getLogger('AbstractBenchmark')


class AbstractBenchmark(abc.ABC, metaclass=abc.ABCMeta):

    def __init__(self, seed: Union[int, np.random.RandomState, None] = None, **kwargs):
        """
        Interface for benchmarks.

        A benchmark consists of two building blocks, the target function and
        the configuration space. Furthermore it can contain additional
        benchmark-specific information such as the location and the function
        value of the global optima.
        New benchmarks should be derived from this base class or one of its
        child classes.

        Parameters
        ----------
        seed: int, np.random.RandomState, None
            The default random state for the benchmark. If type is int, a
            np.random.RandomState with seed `seed` is created. If type is None,
            create a new random state.
        """
        self.seed = seed
        self.configuration_space = self.get_configuration_space(self.seed)

    @abc.abstractmethod
    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Objective function.

        Override this function to provide your benchmark function. This
        function will be called by one of the evaluate functions. For
        flexibility you have to return a dictionary with the only mandatory
        key being `function_value`, the objective function value for the
        `configuration` which was passed. By convention, all benchmarks are
        minimization problems.

        Parameters
        ----------
        configuration : Dict
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        seed : np.random.RandomState, int, None
            It might be useful to pass a `seed` argument to the function call to
            bypass the default "seed" generator. Only using the default random
            state (`self.seed`) could lead to an overfitting towards the
            `self.seed`'s seed.

        Returns
        -------
        Dict
            Must contain at least the key `function_value` and `cost`.
        """
        NotImplementedError()

    @abc.abstractmethod
    def objective_function_test(self, configuration: Union[ConfigSpace.Configuration, Dict],
                                fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                                seed: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        If there is a different objective function for offline testing, e.g
        testing a machine learning on a hold extra test set instead
        on a validation set override this function here.

        Parameters
        ----------
        configuration : Dict
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        seed : np.random.RandomState, int, None
            see :py:func:`~HPOBench.abstract_benchmark.objective_function`

        Returns
        -------
        Dict
            Must contain at least the key `function_value` and `cost`.
        """
        NotImplementedError()

    @staticmethod
    def check_parameters(wrapped_function):
        """
        Wrapper for the objective_function and objective_function_test.
        This function verifies the correctness of the input configuration and the given fidelity.

        It ensures that both, configuration and fidelity, don't contain any wrong parameters or any conditions are
        violated.

        If the argument 'fidelity' is not specified or a single parameter is missing, then the corresponding default
        fidelities are filled in.

        We cast them to a ConfigSpace.Configuration object to ensure that no conditions are violated.
        """

        # Copy all documentation from the underlying function except the annotations.
        @functools.wraps(wrapped=wrapped_function, assigned=('__module__', '__name__', '__qualname__', '__doc__',))
        def wrapper(self, configuration: Union[ConfigSpace.Configuration, Dict],
                     **kwargs):

            configuration = AbstractBenchmark._check_and_cast_configuration(configuration, self.configuration_space)

            # All benchmarks should work on dictionaries. Cast the both objects to dictionaries.
            return wrapped_function(self, configuration.get_dictionary(), **kwargs)
        return wrapper

    @staticmethod
    def _check_and_cast_configuration(configuration: Union[Dict, ConfigSpace.Configuration],
                                      configuration_space: ConfigSpace.ConfigurationSpace) \
            -> ConfigSpace.Configuration:
        """ Helper-function to evaluate the given configuration.
            Cast it to a ConfigSpace.Configuration and evaluate if it violates its boundaries.

            Note:
                We remove inactive hyperparameters from the given configuration. Inactive hyperparameters are
                hyperparameters that are not relevant for a configuration, e.g. hyperparameter A is only relevant if
                hyperparameter B=1 and if B!=1 then A is inactive and will be removed from the configuration.
                Since the authors of the benchmark removed those parameters explicitly, they should also handle the
                cases that inactive parameters are not present in the input-configuration.
        """

        if isinstance(configuration, dict):
            configuration = ConfigSpace.Configuration(configuration_space, configuration,
                                                      allow_inactive_with_values=True)
        elif isinstance(configuration, ConfigSpace.Configuration):
            configuration = configuration
        else:
            raise TypeError(f'Configuration has to be from type List, np.ndarray, dict, or '
                            f'ConfigSpace.Configuration but was {type(configuration)}')

        all_hps = set(configuration_space.get_hyperparameter_names())
        active_hps = configuration_space.get_active_hyperparameters(configuration)
        inactive_hps = all_hps - active_hps

        if len(inactive_hps) != 0:
            logger.debug(f'There are inactive {len(inactive_hps)} hyperparameter: {inactive_hps}'
                         'Going to remove them from the configuration.')

        configuration = deactivate_inactive_hyperparameters(configuration, configuration_space)
        configuration_space.check_configuration(configuration)

        return configuration


    def __call__(self, configuration: Dict, **kwargs) -> float:
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']

    @staticmethod
    @abc.abstractmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        """ Defines the configuration space for each benchmark.
        Parameters
        ----------
        seed: int, None
            Seed for the configuration space.

        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's parameters
        """
        raise NotImplementedError()



