from typing import Dict, List, Optional, Union

import ConfigSpace as CS

import numpy as np

from External.parzen_estimator import (
    CategoricalParzenEstimator,
    NumericalParzenEstimator,
    build_categorical_parzen_estimator,
    build_numerical_parzen_estimator,
)
from External.parzen_estimator.constants import config2type
from External.parzen_estimator.loglikelihoods import compute_config_loglikelihoods
from External.parzen_estimator.utils import exp

from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import Sobol


SAMPLING_CHOICES = {"sobol": Sobol, "lhs": LHS}
ParzenEstimatorType = Union[CategoricalParzenEstimator, NumericalParzenEstimator]
SampleDataType = Union[List[np.ndarray], np.ndarray, Dict[str, np.ndarray]]


def _validate_weights(weights_array: np.ndarray) -> None:
    if not np.all(np.all(weights_array == weights_array[0], axis=0)):
        raise ValueError("weights for all Parzen estimators must be same. Check if the same weight_func is used.")


def _validate_size(parzen_estimators: Dict[str, ParzenEstimatorType], size: int) -> None:
    if any(pe.size != size for pe in parzen_estimators.values()):
        raise ValueError("All parzen estimators must be identical.")


class MultiVariateParzenEstimator:
    def __init__(self, parzen_estimators: Dict[str, ParzenEstimatorType]):
        """
        MultiVariateParzenEstimator.

        Attributes:
            parzen_estimators (Dict[str, ParzenEstimatorType]):
                Parzen estimators for each hyperparameter.
            dim (int):
                The dimensions of search space.
            size (int):
                The number of observations used for the parzen estimators.
            weights (np.ndarray):
                The weight values for each basis.
        """
        self._parzen_estimators = parzen_estimators
        self._size = list(parzen_estimators.values())[0].size
        _validate_size(parzen_estimators, size=self._size)
        _validate_weights(weights_array=np.asarray([pe._weights for pe in parzen_estimators.values()]))
        self._param_names = list(parzen_estimators.keys())
        self._dim = len(parzen_estimators)
        self._weights = parzen_estimators[self._param_names[0]]._weights.copy()
        self._hypervolume = np.prod([float(pe.domain_size) for pe in parzen_estimators.values()])

    def __repr__(self) -> str:
        return "\n".join(
            [f"({idx + 1}): {hp_name}\n{pe}" for idx, (hp_name, pe) in enumerate(self._parzen_estimators.items())]
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.pdf(x)

    def __len__(self) -> int:
        return self._dim

    def __getitem__(self, key: str) -> ParzenEstimatorType:
        return self._parzen_estimators[key]

    def __contains__(self, key: str) -> bool:
        return key in self._parzen_estimators

    def _convert_X_dict_to_X_list(self, X: SampleDataType) -> List[np.ndarray]:
        return X if isinstance(X, list) else [X[param_name] for param_name in self._param_names]

    def _convert_X_list_to_X_dict(self, X: SampleDataType) -> Dict[str, np.ndarray]:
        return X if isinstance(X, dict) else {param_name: X[dim] for dim, param_name in enumerate(self._param_names)}

    def dimension_wise_pdf(self, X: SampleDataType, return_dict: bool = False) -> SampleDataType:
        """
        Compute the probability density value in each dimension given data points X.

        Args:
            X (SampleDataType):
                Data points with the shape of (dim, n_samples)
            return_dict (bool):
                Whether the return should be dict or list.

        Returns:
            pdf_values (SampleDataType):
                The density values in each dimension for each data point.
                The shape is (dim, n_samples).
        """
        _X = self._convert_X_dict_to_X_list(X)
        n_samples = _X[0].size
        pdfs = np.zeros((self._dim, n_samples))

        for dim, param_name in enumerate(self._param_names):
            pe = self._parzen_estimators[param_name]
            pdfs[dim] += pe.pdf(_X[dim])

        return self._convert_X_list_to_X_dict(pdfs) if return_dict else pdfs

    def log_pdf(self, X: SampleDataType) -> np.ndarray:
        """
        Compute the probability density value given data points X.

        Args:
            X (SampleDataType):
                Data points with the shape of (dim, n_samples)

        Returns:
            log_pdf_values (np.ndarray):
                The log density values for each data point.
                The shape is (n_samples, ).
        """
        _X = self._convert_X_dict_to_X_list(X)
        n_samples = _X[0].size
        blls = np.zeros((self._dim, self._size, n_samples))

        for d, (hp_name, pe) in enumerate(self._parzen_estimators.items()):
            blls[d] += pe.basis_loglikelihood(_X[d])

        config_ll = compute_config_loglikelihoods(blls, self._weights)
        return config_ll

    def pdf(self, X: SampleDataType) -> np.ndarray:
        """
        Compute the probability density value given data points X.

        Args:
            X (SampleDataType):
                Data points with the shape of (dim, n_samples)

        Returns:
            pdf_values (np.ndarray):
                The density values for each data point.
                The shape is (n_samples, ).
        """
        return exp(self.log_pdf(X))

    def sample(
        self,
        n_samples: int,
        rng: np.random.RandomState,
        dim_independent: bool = False,
        return_dict: bool = False,
    ) -> SampleDataType:
        samples = []
        if dim_independent:
            samples = [pe.sample(rng, n_samples) for d, pe in enumerate(self._parzen_estimators.values())]
        else:
            indices = rng.choice(self._size, p=self._weights, size=n_samples)
            samples = [pe.sample_by_indices(rng, indices) for d, pe in enumerate(self._parzen_estimators.values())]

        return self._convert_X_list_to_X_dict(samples) if return_dict else samples

    def uniform_sample(
        self,
        n_samples: int,
        rng: np.random.RandomState,
        sampling_method: str = "sobol",  # Literal["sobol", "lhs"]
        return_dict: bool = False,
    ) -> SampleDataType:
        """
        Sample points using latin hypercube sampling.

        Args:
            parzen_estimator (List[ParzenEstimatorType]):
                The list that contains the information of each dimension.
            n_samples (int):
                The number of samples.
            return_dict (bool):
                Whether the return should be dict or list.

        Returns:
            samples (SampleDataType):
                Random samplings converted accordingly.
                The shape is (dim, n_samples).
        """
        if sampling_method not in SAMPLING_CHOICES:
            raise ValueError(f"sampling_method must be in {SAMPLING_CHOICES}, but got {sampling_method}")

        sampler = SAMPLING_CHOICES[sampling_method](d=self._dim, seed=rng)
        # We need to do it to maintain dtype for each dimension
        samples = [sample for sample in sampler.random(n=n_samples).T]

        for d, pe in enumerate(self._parzen_estimators.values()):
            samples[d] = pe.uniform_to_valid_range(samples[d])

        return self._convert_X_list_to_X_dict(samples) if return_dict else samples

    @property
    def size(self) -> int:
        return self._size

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def param_names(self) -> List[str]:
        return self._param_names[:]

    @property
    def hypervolume(self) -> float:
        return self._hypervolume


def get_multivar_pdf(
    observations: Dict[str, np.ndarray],
    config_space: CS.ConfigurationSpace,
    *,
    default_min_bandwidth_factor: float = 1e-2,
    prior: bool = True,
    weights: Optional[np.ndarray] = None,
    vals_for_categorical_is_indices: bool = False,
) -> MultiVariateParzenEstimator:

    hp_names = config_space.get_hyperparameter_names()
    parzen_estimators: Dict[str, ParzenEstimatorType] = {}

    for hp_name in hp_names:
        config = config_space.get_hyperparameter(hp_name)
        config_type = config.__class__.__name__
        is_ordinal = config_type.startswith("Ordinal")
        is_categorical = config_type.startswith("Categorical")
        kwargs = dict(vals=observations[hp_name], config=config, prior=prior, weights=weights)

        if is_categorical:
            kwargs.update(vals_is_indices=vals_for_categorical_is_indices)
            parzen_estimators[hp_name] = build_categorical_parzen_estimator(**kwargs)
        else:
            kwargs.update(
                dtype=config2type[config_type],
                is_ordinal=is_ordinal,
                default_min_bandwidth_factor=default_min_bandwidth_factor,
            )
            parzen_estimators[hp_name] = build_numerical_parzen_estimator(**kwargs)

    return MultiVariateParzenEstimator(parzen_estimators)


def over_resample(
    config_space: CS.ConfigurationSpace,
    observations: Dict[str, np.ndarray],
    n_resamples: int,
    rng: np.random.RandomState,
    default_min_bandwidth_factor: float = 1e-1,
    prior: bool = False,
    dim_independent_resample: bool = False,
    weights: Optional[np.ndarray] = None,
) -> MultiVariateParzenEstimator:
    mvpdf = get_multivar_pdf(
        observations=observations,
        config_space=config_space,
        default_min_bandwidth_factor=default_min_bandwidth_factor,
        prior=prior,
        weights=weights,
    )
    resampled_configs = {
        param_name: samples
        for param_name, samples in zip(
            mvpdf.param_names, mvpdf.sample(n_samples=n_resamples, rng=rng, dim_independent=dim_independent_resample)
        )
    }
    return get_multivar_pdf(
        observations=resampled_configs,
        config_space=config_space,
        default_min_bandwidth_factor=default_min_bandwidth_factor,
        prior=False,
        vals_for_categorical_is_indices=True,
        weights=weights,
    )
