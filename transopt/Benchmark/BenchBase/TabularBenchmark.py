import os
import json
import logging
import ConfigSpace
import numpy as np
import ConfigSpace as CS
from typing import Union, Dict, List
from urllib.parse import urlparse
from pathlib import Path
from transopt.Benchmark.BenchBase import BenchmarkBase
from transopt.utils.Read import read_file

logger = logging.getLogger("TabularBenchmark")



class TabularBenchmark(BenchmarkBase):
    def __init__(
            self,
            task_name: str,
            task_type: str,
            budget: int,
            workload,
            path: str = None,
            seed: Union[int, np.random.RandomState, None] = None,
            space_info: Dict = None,
            **kwargs,
    ):

        self.path = path

        parsed = urlparse(path)
        if parsed.scheme and parsed.netloc:
            return "URL"
        # If the string is a valid file path
        elif os.path.exists(path) or os.path.isabs(path):
            dir_path = Path(path)
            workload_path = dir_path / workload
            data = read_file(workload_path)
            unnamed_columns = [col for col in data.columns if "Unnamed" in col]
            # delete the unnamed column
            data.drop(unnamed_columns, axis=1, inplace=True)

            para_names = [value for value in data.columns]
            if space_info is None or not isinstance(data, dict):
                self.space_info = {}
            else:
                self.space_info = space_info

            if 'input_dim' not in self.space_info and 'num_objective' not in self.space_info:
                self.space_info['input_dim'] = len(para_names) - 1
                self.input_dim = self.space_info['input_dim']
                self.space_info['num_objective'] = len(para_names) - self.space_info['input_dim']
                self.num_objective = self.space_info['num_objective']
            else:
                if 'num_objective' in self.space_info:
                    self.space_info['input_dim'] = len(para_names) - self.space_info['num_objective']
                    self.input_dim = self.space_info['input_dim']

                if 'input_dim' in self.space_info:
                    self.space_info['num_objective'] = len(para_names) - self.space_info['input_dim']
                    self.num_objective = self.space_info['num_objective']



            if 'variables' not in self.space_info:
                self.space_info['variables'] = {}
                for i in range(self.space_info['input_dim']):
                    var_name = para_names[i]
                    max_value = data[var_name].max()
                    min_value = data[var_name].min()
                    contains_decimal = False
                    contains_str = False
                    if data[var_name][1:].nunique() > 10:
                        for item in data[var_name][1:]:
                            if isinstance(item, str):
                                contains_str = True
                            if int(item) - item != 0:
                                contains_decimal = True
                                break  # 如果找到小数，无需继续检查
                        if contains_decimal:
                            var_type =  'continuous'
                            self.space_info['variables'][var_name] = {'bounds': [min_value, max_value],
                                                                      'type': var_type}
                        elif contains_str:
                            var_type = 'categorical'
                            self.space_info['variables'][var_name] = {'bounds': list(data[var_name][1:].unique()),
                                                                      'type': var_type}
                        else:
                            var_type = 'discrete'
                            self.space_info['variables'][var_name] = {'bounds': [min_value, max_value],
                                                                      'type': var_type}

                    else:
                        var_type = 'categorical'
                        self.space_info['variables'][var_name] = {'bounds': list(data[var_name][1:].unique()),
                                                                  'type': var_type}

            data["fitness"] = data[para_names[-1]]
            data['config'] = data.apply(lambda row: row.tolist(), axis=1)
            data["config_s"] = data["config"].astype(str)
        else:
            raise ValueError("Unknown path type, only accept url or file path")

        super(TabularBenchmark, self).__init__(seed, **kwargs)
        self.var_range = self.get_configuration_bound()
        self.var_type = self.get_configuration_type()
        self.data_set = data

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

    def sample_dataframe(key, df, p_remove=0.):
        """Randomly sample dataframe by the removal percentage."""
        if p_remove < 0 or p_remove >= 1:
            raise ValueError(
                f'p_remove={p_remove} but p_remove must be <1 and >= 0.')
        if p_remove > 0:
            n_remain = (1 - p_remove) * len(df)
            n_remain = int(np.ceil(n_remain))
            df = df.sample(n=n_remain, replace=False, random_state=key[0])
        return df

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

    def get_configuration_space(
            self, seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:
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
        variables = []

        for k,v in self.space_info['variables'].items():
            lower = v['bounds'][0]
            upper = v['bounds'][1]
            if 'continuous' == v['type']:
                variables.append(CS.UniformFloatHyperparameter(k, lower=lower, upper=upper))
            elif 'discrete' == v['type']:
                variables.append(CS.UniformIntegerHyperparameter(k, lower=lower, upper=upper))
            elif 'categorical' == v['type']:
                variables.append(CS.CategoricalHyperparameter(k, choices=v['bounds']))
            else:
                raise ValueError('Unknown variable type')

        cs.add_hyperparameters(variables)
        return cs

    def get_fidelity_space(
            self, seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:
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
        return {}


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
