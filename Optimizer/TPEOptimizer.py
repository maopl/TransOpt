from typing import Any, Dict, List, Optional

import ConfigSpace as CS

import numpy as np

from External.tpe.optimizer.tpe import TreeStructuredParzenEstimator
from External.tpe.utils.constants import PercentileFuncMaker, default_percentile_maker
from abc import ABCMeta, abstractmethod

from External.tpe.utils.constants import NumericType
from External.tpe.utils.utils import get_logger, get_random_sample, revert_eval_config, store_results


class TPEOptimizer(metaclass=ABCMeta):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        resultfile: str = "temp",
        n_init: int = 10,
        max_evals: int = 100,
        seed: Optional[int] = None,
        metric_name: str = "loss",
        runtime_name: str = "iter_time",
        only_requirements: bool = False,
        n_ei_candidates: int = 24,
        # TODO: task names for transfer learning
        result_keys: List[str] = ["loss"],
        min_bandwidth_factor: float = 1e-1,
        top: float = 1.0,
        # TODO: Make dict of percentile_func_maker
        percentile_func_maker: PercentileFuncMaker = default_percentile_maker,
    ):
        """
        Args:
            obj_func (ObjectiveFunc): The objective function.
            config_space (CS.ConfigurationSpace): The searching space of the task
            resultfile (str): The name of the result file to output in the end
            n_init (int): The number of random sampling before using TPE
            max_evals (int): The number of total evaluations.
            seed (int): The random seed.
            metric_name (str): The name of the metric (or that of the objective function value)
            runtime_name (str): The name of the runtime metric.
            only_requirements (bool): If True, we only save runtime and loss.
            n_ei_candidates (int): The number of samplings to optimize the EI value
            result_keys (List[str]): Keys of results.
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            top (float): The hyperparam of the cateogircal kernel. It defines the prob of the top category.
        """

        self._rng = np.random.RandomState(seed)
        self._n_init, self._max_evals = n_init, max_evals
        self.resultfile = resultfile
        self._hp_names = list(config_space._hyperparameters.keys())
        self._metric_name = metric_name
        self._runtime_name = runtime_name
        self._result_keys = result_keys[:]
        self._requirements = [metric_name, self._runtime_name] if only_requirements else None

        self._config_space = config_space
        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "CategoricalHyperparameter"
            for hp_name in self._hp_names
        }
        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "OrdinalHyperparameter"
            for hp_name in self._hp_names
        }

        self._tpe_samplers = {
            key: TreeStructuredParzenEstimator(
                config_space=config_space,
                metric_name=key,
                runtime_name=runtime_name,
                n_ei_candidates=n_ei_candidates,
                seed=seed,
                min_bandwidth_factor=min_bandwidth_factor,
                top=top,
                percentile_func=percentile_func_maker(),
            )
            for key in result_keys
        }

    def update(self, eval_config: Dict[str, Any], results: Dict[str, float], runtime: float) -> None:
        for key, val in results.items():
            self._tpe_samplers[key].update_observations(eval_config=eval_config, loss=val, runtime=runtime)

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        observations = self._tpe_samplers[self._metric_name].observations
        for key in self._result_keys:
            observations[key] = self._tpe_samplers[key].observations[key]

        return observations

    def _get_config_cands(self, n_samples_dict: Dict[str, int]) -> Dict[str, np.ndarray]:
        config_cands: Dict[str, np.ndarray] = {}
        for key in self._result_keys:
            tpe_sampler = self._tpe_samplers[key]
            n_samples = n_samples_dict.get(key, tpe_sampler._n_ei_candidates)
            if n_samples == 0:
                continue

            configs = tpe_sampler.get_config_candidates(n_samples)
            if len(config_cands) == 0:
                config_cands = configs
            else:
                config_cands = {
                    hp_name: np.concatenate([config_cands[hp_name], configs[hp_name]]) for hp_name in configs.keys()
                }

        return config_cands

    def _compute_probability_improvement(
        self, config_cands: Dict[str, np.ndarray], weight_dict: Dict[str, float]
    ) -> np.ndarray:
        pi_config_array = np.zeros((len(self._result_keys), config_cands[self._hp_names[0]].size))
        weights = np.ones(len(self._result_keys))

        for i, key in enumerate(self._result_keys):
            tpe_sampler = self._tpe_samplers[key]
            pi_config_array[i] += tpe_sampler.compute_probability_improvement(config_cands=config_cands)
            weights[i] = weight_dict[key]

        return weights @ pi_config_array

    def sample(
        self, weight_dict: Optional[Dict[str, float]] = None, n_samples_dict: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Sample a configuration using tree-structured parzen estimator (TPE)

        Args:
            weights (Optional[Dict[str, float]]):
                Weights for each tpe samplers.
            n_samples_dict (Optional[Dict[str, int]]):
                The number of samples for each tpe samplers.

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration from TPE
        """
        n_samples_dict = {} if n_samples_dict is None else n_samples_dict
        weight_dict = {key: 1.0 for key in self._result_keys} if weight_dict is None else weight_dict

        config_cands = self._get_config_cands(n_samples_dict)
        pi_config = self._compute_probability_improvement(config_cands, weight_dict)
        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: config_cands[hp_name][best_idx] for dim, hp_name in enumerate(self._hp_names)}

        return self._revert_eval_config(eval_config=eval_config)

    def sample_from_tabular(self, X:np.ndarray, idxs):
        weight_dict = {key: 1.0 for key in self._result_keys}

        num, dim = X.shape
        config_cands = {}
        for d in range(dim):
            config_cands[f'x{d}'] = X[:,d]
        pi_config = self._compute_probability_improvement(config_cands, weight_dict)
        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: config_cands[hp_name][best_idx] for dim, hp_name in enumerate(self._hp_names)}

        return self._revert_eval_config(eval_config=eval_config), idxs[best_idx]


    def initial_sample(self) -> Dict[str, Any]:
        """
        Sampling Method up to n_init configurations

        Returns:
            samples (Dict[str, Any]):
                Typically randomly sampled configurations

        """
        eval_config = {hp_name: self._get_random_sample(hp_name=hp_name) for hp_name in self._hp_names}
        return self._revert_eval_config(eval_config=eval_config)

    def _get_random_sample(self, hp_name: str) -> NumericType:
        return get_random_sample(
            hp_name=hp_name,
            rng=self._rng,
            config_space=self._config_space,
            is_categorical=self._is_categoricals[hp_name],
            is_ordinal=self._is_ordinals[hp_name],
        )

    def _revert_eval_config(self, eval_config: Dict[str, NumericType]) -> Dict[str, Any]:
        return revert_eval_config(
            eval_config=eval_config,
            config_space=self._config_space,
            is_categoricals=self._is_categoricals,
            is_ordinals=self._is_ordinals,
            hp_names=self._hp_names,
        )