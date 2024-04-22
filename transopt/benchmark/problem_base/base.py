""" Base-class of all benchmarks """

import abc
import functools
import logging
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace
import numpy as np
from typing import Union, Dict

from ConfigSpace.util import deactivate_inactive_hyperparameters

logger = logging.getLogger("AbstractProblem")


class ProblemBase(abc.ABC, metaclass=abc.ABCMeta):
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
            np.random.RandomState with seed `rng` is created. If type is None,
            create a new random state.
        """

        self.seed = seed
        self.configuration_space = self.get_configuration_space(self.seed)
        self.fidelity_space = self.get_fidelity_space(self.seed)
        self.objective_info = self.get_objectives(self.seed)
        
        self.input_dim = len(self.configuration_space.keys())
        self.num_objective = len(self.objective_info)

    def f(self, configuration, fidelity=None, seed=None, **kwargs) -> Dict:
        # Check validity of configuration and fidelity before evaluation
        self.check_validity(configuration, fidelity)

        # Delegate to the specific evaluation method implemented by subclasses
        return self.objective_function(configuration, fidelity, seed, **kwargs)

    @abc.abstractmethod
    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        """Implement this method in subclasses to define specific evaluation logic."""
        raise NotImplementedError

    def check_validity(self, configuration, fidelity):
        # Check if each configuration key and value is valid
        for key, value in configuration.items():
            if key not in self.configuration_space.ranges:
                raise ValueError(f"Configuration key {key} is not valid.")

            range = self.configuration_space.ranges[key]
            if not (range[0] <= value <= range[1]):
                raise ValueError(
                    f"Value of {key}={value} is out of allowed range {range}."
                )

        if fidelity is None:
            return

        # Check if each fidelity key and value is valid
        for key, value in fidelity.items():
            if key not in self.fidelity_space.ranges:
                raise ValueError(f"Fidelity key {key} is not valid.")
            range = self.fidelity_space.ranges[key]
            if not (range[0] <= value <= range[1]):
                raise ValueError(
                    f"Value of {key}={value} is out of allowed range {range}."
                )

    def __call__(self, configuration: Dict, **kwargs) -> float:
        """Provides interface to use, e.g., SciPy optimizers"""
        return self.f(configuration, **kwargs)["function_value"]

    @abc.abstractmethod
    def get_configuration_space(self, seed: Union[int, None] = None) -> SearchSpace:
        """Defines the configuration space for each benchmark.
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

    @abc.abstractmethod
    def get_fidelity_space(self, seed: Union[int, None] = None) -> FidelitySpace:
        """Defines the available fidelity parameters as a "fidelity space" for each benchmark.
        Parameters
        ----------
        seed: int, None
            Seed for the fidelity space.
        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's fidelity parameters
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_objectives(self, seed: Union[int, None] = None) -> list:
        """Defines the available fidelity parameters as a "fidelity space" for each benchmark.
        Parameters
        ----------
        seed: int, None
            Seed for the fidelity space.
        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's fidelity parameters
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def get_meta_information() -> Dict:
        """Provides some meta information about the benchmark.

        Returns
        -------
        Dict
            some human-readable information

        """
        raise NotImplementedError()