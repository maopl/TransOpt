""" Base-class of configuration optimization benchmarks """

import json
import logging
import ConfigSpace
import numpy as np
from typing import Union, Dict, List

from transopt.Benchmark.BenchBase import BenchmarkBase

logger = logging.getLogger("ConfigOptBenchmark")


class NonTabularOptBenchmark(BenchmarkBase):
    def __init__(
        self,
        task_name: str,
        task_type: str,
        budget: int,
        workload,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.task_name = task_name
        self.budget = budget
        self.workload = workload
        self.lock_flag = False

        super(NonTabularOptBenchmark, self).__init__(seed, **kwargs)
        self.var_range = self.get_configuration_bound()
        self.var_type = self.get_configuration_type()
        self.input_dim = len(self.configuration_space.keys())
        self.num_objective = self.get_meta_information()["number_objective"]

    def f(
        self,
        configuration: Union[ConfigSpace.Configuration, Dict],
        fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
        **kwargs,
    ) -> Dict:
        results = self.objective_function(
            configuration=configuration,
            fidelity=fidelity,
            seed=self.seed,
            kwargs=kwargs,
        )
        return results

    def get_configuration_bound(self):
        configuration_bound = {}
        for k, v in self.configuration_space.items():
            if type(v) is ConfigSpace.CategoricalHyperparameter:
                configuration_bound[k] = [0, len(v.choices) - 1]
            else:
                configuration_bound[k] = [v.lower, v.upper]

        return configuration_bound

    def get_configuration_type(self):
        configuration_type = {}
        for k, v in self.configuration_space.items():
            configuration_type[k] = type(v).__name__
        return configuration_type

    def get_budget(self) -> int:
        """Provides the function evaluations number about the benchmark.

        Returns
        -------
        int
            some human-readable information

        """
        return self.budget

    def get_name(self) -> str:
        """Provides the task name about the benchmark.

        Returns
        -------
        str
            some human-readable information

        """
        return self.task_name

    def get_type(self) -> str:
        """Provides the task type about the benchmark.

        Returns
        -------
        str
            some human-readable information

        """
        return self.task_type

    def get_input_dim(self) -> int:
        """Provides the input dimension about the benchmark.

        Returns
        -------
        int
            some human-readable information

        """
        return self.input_dim

    def get_objective_num(self) -> int:
        return self.num_objective

    def lock(self):
        self.lock_flag = True

    def unlock(self):
        self.lock_flag = False

    def get_lock_state(self) -> bool:
        return self.lock_flag


class TabularOptBenchmark(NonTabularOptBenchmark):
    def __init__(
        self,
        task_name: str,
        task_type: str,
        budget: int,
        workload,
        path: str = None,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ):

        self.path = path
        self.data_set = None
        super(TabularOptBenchmark, self).__init__(
            task_name, task_type, budget, workload, seed, **kwargs
        )

    def f(
        self,
        configuration: Union[ConfigSpace.Configuration, Dict, None],
        fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
        **kwargs,
    ) -> Dict:
        if "idx" not in kwargs:
            raise ValueError("The passed arguments must include the 'idx' parameter.")
        idx = kwargs["idx"]
        results = self.objective_function(
            configuration={}, fidelity={}, seed=self.seed, idx=idx
        )
        return results

    def objective_function(
        self,
        configuration: Union[ConfigSpace.Configuration, Dict],
        fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        pass

    def get_configuration_space(
        self, seed: Union[int, None] = None
    ) -> ConfigSpace.ConfigurationSpace:
        pass

    def get_fidelity_space(
        self, seed: Union[int, None] = None
    ) -> ConfigSpace.ConfigurationSpace:
        pass
    def get_meta_information() -> Dict:
        pass

    def read_data_from_json(self):
        if not self.data_path.endswith(".json"):
            logger.error("Not a json file.")
            raise NameError("Not a json file.")

        try:
            with open(self.data_path, "r") as f:
                data_set = json.load(f)
            return data_set
        except json.JSONDecodeError:
            logger.error(f"Can not parse the json file {self.data_path}.")
            return None

    def get_dataset_size(self):
        raise NotImplementedError

    def get_var_by_idx(self, idx):
        raise NotImplementedError

    def get_idx_by_var(self, vectors):
        raise NotImplementedError

    def get_unobserved_vars(self):
        raise NotImplementedError

    def get_unobserved_idxs(self):
        raise NotImplementedError
