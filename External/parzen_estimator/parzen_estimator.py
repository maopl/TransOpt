from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Type, Union
from typing_extensions import Literal

import numpy as np

from External.parzen_estimator.constants import (
    CategoricalHPType,
    EPS,
    NULL_VALUE,
    NumericType,
    NumericalHPType,
    SQR2,
    SQR2PI,
    uniform_weight,
)
from External.parzen_estimator.utils import (
    _get_min_bandwidth_factor,
    calculate_norm_consts,
    erf,
    exp,
    log,
    validate_and_update_dtype,
    validate_and_update_q,
)


HYPEROPT = "hyperopt"
OPTUNA = "optuna"


class AbstractParzenEstimator(metaclass=ABCMeta):
    _weights: np.ndarray
    _dtype: Type[np.number]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.pdf(x)

    @abstractmethod
    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        """
        Convert the uniform samples [0, 1] into valid range.

        Args:
            x (np.ndarray):
                uniform samples in [0, 1].
                The shape is (n_samples, ).

        Returns:
            converted_x (np.ndarray):
                Converted values.
                The shape is (n_samples, ).
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability density values for each data point.

        Args:
            x (np.ndarray): The sampled values to compute density values
                            The shape is (n_samples, )

        Returns:
            pdf_vals (np.ndarray):
                The density values given sampled values
                The shape is (n_samples, )
        """
        raise NotImplementedError

    @abstractmethod
    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the kernel value for each basis in the parzen estimator.

        Args:
            x (np.ndarray): The sampled values to compute each kernel value
                            The shape is (n_samples, )

        Returns:
            basis_loglikelihoods (np.ndarray):
                The kernel values for each basis given sampled values
                The shape is (B, n_samples)
                where B is the number of basis and n_samples = xs.size

        NOTE:
            When the parzen estimator is computed by:
                p(x) = sum[i = 0 to B] weights[i] * basis[i](x)
            where basis[i] is the i-th kernel function.
            Then this function returns the following:
                [log(basis[0](xs)), ..., log(basis[B - 1](xs))]
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def domain_size(self) -> NumericType:
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    def _validate_weights(self) -> None:
        assert isinstance(self._weights, np.ndarray)  # mypy redefinition
        if self._weights.size != self.size:
            raise ValueError(
                f"The size of weights must be {self.size}, but got {self._weights.size}. "
                "Check if including the weight for prior."
            )
        if not np.isclose(self._weights.sum(), 1.0):
            raise ValueError(f"The sum of weights must be 1, but got {self._weights.sum()}")


class NumericalParzenEstimator(AbstractParzenEstimator):
    _HEURISTICS = [HYPEROPT, OPTUNA]

    def __init__(
        self,
        samples: np.ndarray,
        lb: NumericType,
        ub: NumericType,
        *,
        q: Optional[NumericType] = None,
        hard_lb: Optional[NumericType] = None,
        hard_ub: Optional[NumericType] = None,
        dtype: Type[Union[np.number, int, float]] = np.float64,
        compress: bool = False,
        min_bandwidth_factor: float = 1e-2,
        magic_clip: bool = False,
        magic_clip_exponent: float = 1.0,
        prior: bool = True,
        weights: Optional[np.ndarray] = None,
        heuristic: Optional[Literal["hyperopt", "optuna"]] = None,
        space_dim: Optional[int] = None,
    ):

        self._lb, self._ub, self._q = lb, ub, q
        self._hard_lb: NumericType = hard_lb if hard_lb is not None else lb
        self._hard_ub: NumericType = hard_ub if hard_ub is not None else ub
        self._size = samples.size + prior
        self._heuristic = heuristic
        self._space_dim = space_dim
        self._weights = weights.copy() if weights is not None else uniform_weight(samples.size + prior)
        self._dtype: Type[np.number]
        self._validate(dtype, samples)

        self._means: np.ndarray
        self._stds: np.ndarray
        self._logpdf_consts: np.ndarray
        self._norm_consts: np.ndarray
        self._index_to_basis_index: np.ndarray = np.arange(samples.size + prior)

        magic_factor = 1.0 / self._size ** magic_clip_exponent
        min_bandwidth_factor = max(min_bandwidth_factor, magic_factor) if magic_clip else min_bandwidth_factor
        self._calculate(samples=samples, min_bandwidth_factor=min_bandwidth_factor, prior=prior, compress=compress)

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}(\n\tlb={self.lb}, ub={self.ub}, q={self.q},\n"
        for i, (m, s, w) in enumerate(zip(self._means, self._stds, self._weights)):
            ret += f"\t({i + 1}) weight: {w}, basis: GaussKernel(mean={m}, std={s}),\n"
        return ret + ")"

    def _validate_discrete_info(self) -> None:
        if self._q is None:
            # Continuous
            return

        if self.lb == self._hard_lb or self.ub == self._hard_ub:
            self._lb -= 0.5 * self._q
            self._ub += 0.5 * self._q

    def _validate(self, dtype: Type[Union[np.number, int, float]], samples: np.ndarray) -> None:
        self._validate_weights()
        self._dtype = validate_and_update_dtype(dtype=dtype)
        self._q = validate_and_update_q(dtype=self._dtype, q=self.q)
        q = self.q
        if np.any(samples < self._hard_lb) or np.any(samples > self._hard_ub):
            raise ValueError(f"All the samples must be in [{self._hard_lb}, {self._hard_ub}].")
        if q is not None:
            valid_vals = np.linspace(self._hard_lb, self._hard_ub, self.domain_size)
            cands = np.unique(samples)
            converted_cands = np.round((cands - self._hard_lb) / q) * q + self._hard_lb
            if not np.allclose(cands, converted_cands):
                raise ValueError(
                    "All the samples for q != None must be discritized appropriately."
                    f" Expected each value to be in {valid_vals}, but got {cands}"
                )

        self._validate_discrete_info()

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        if self.q is None:
            mahalanobis = ((x - self._means[:, np.newaxis]) / self._stds[:, np.newaxis]) ** 2
            return self._logpdf_consts[:, np.newaxis] - 0.5 * mahalanobis
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x - 0.5 * self.q, self.lb))
            return log(integral_u - integral_l + EPS)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.q is None:
            norm_consts = self._norm_consts / (SQR2PI * self._stds)  # noqa: F841
            mahalanobis = ((x[:, np.newaxis] - self._means) / self._stds) ** 2  # noqa: F841
            return np.sum(self._weights * norm_consts * exp(-0.5 * mahalanobis), axis=-1)
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x - 0.5 * self.q, self.lb))
            return np.sum(self._weights[:, np.newaxis] * (integral_u - integral_l), axis=0)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative density function values.

        Args:
            x (np.ndarray): Samples to compute the cdf

        Returns:
            cdf (np.ndarray):
                The cumulative density function value for each sample
                cdf[i] = integral[from -inf to x[i]] pdf(x') dx'
        """
        z = (x - self._means[:, np.newaxis]) / (SQR2 * self._stds[:, np.newaxis])
        norm_consts = self._norm_consts[:, np.newaxis]
        return norm_consts * 0.5 * (1.0 + erf(z))

    def _sample(self, rng: np.random.RandomState, idx: int) -> NumericType:
        idx = self._index_to_basis_index[idx]
        while True:
            val = rng.normal(loc=self._means[idx], scale=self._stds[idx])
            if self._hard_lb <= val <= self._hard_ub:
                return val if self.q is None else np.round((val - self._hard_lb) / self.q) * self.q + self._hard_lb

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        samples = [self._sample(rng, active) for active in rng.choice(self.size, p=self._weights, size=n_samples)]
        return np.array(samples, dtype=self._dtype)

    def sample_by_indices(self, rng: np.random.RandomState, indices: np.ndarray) -> np.ndarray:
        return np.array([self._sample(rng, idx) for idx in indices])

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = self._hard_lb + x * (self._hard_ub - self._hard_lb)
        scaled_x = (
            scaled_x if self.q is None else np.round((scaled_x - self._hard_lb) / self.q) * self.q + self._hard_lb
        )
        return scaled_x.astype(self._dtype)

    def _preproc(self, samples: np.ndarray, prior: bool) -> Tuple[np.ndarray, float, float]:
        means = np.append(samples, 0.5 * (self._hard_lb + self._hard_ub)) if prior else samples.copy()
        std = means.std(ddof=int(means.size > 1))
        IQR = np.subtract.reduce(np.percentile(means, [75, 25]))
        return means, std, IQR

    def _preproc_with_compress(self, samples: np.ndarray, prior: bool) -> Tuple[np.ndarray, float, float]:
        center = 0.5 * (self._hard_lb + self._hard_ub)
        means, invs, counts = np.unique(samples, return_counts=True, return_inverse=True)

        self._index_to_basis_index = np.append(invs, counts.size) if prior else invs
        means = np.append(means, center) if prior else means
        counts = np.append(counts, 1) if prior else counts
        size = np.sum(counts)
        self._weights = counts / size

        mu = (means @ counts) / size
        std = np.sqrt((means - mu) ** 2 @ counts / max(1, size - 1))
        cum_counts = np.cumsum(counts)
        idx_q25 = np.searchsorted(cum_counts, size // 4)
        idx_q75 = np.searchsorted(cum_counts, size * 3 // 4)
        IQR = means[idx_q75] - means[idx_q25]
        return means, std, IQR

    def _bandwidth_heuristic_hyperopt(self, samples: np.ndarray, prior: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the bandwidth based on the 2011 version of TPE.

        Reference:
            J. Bergstra et al. (2011) Algorithms for hyper-parameter optimization.
        """
        means = np.append(samples, 0.5 * (self._hard_lb + self._hard_ub)) if prior else samples.copy()
        order = np.argsort(means)

        sorted_means = np.empty(means.size + 2, dtype=np.float64)
        sorted_means[1:-1] = means[order]
        sorted_means[0], sorted_means[-1] = self._hard_lb, self._hard_ub

        sorted_bandwidth = np.maximum(
            sorted_means[1:-1] - sorted_means[:-2],
            sorted_means[2:] - sorted_means[1:-1],
        )
        return means, sorted_bandwidth[np.argsort(order)]

    def _bandwidth_heuristic_optuna(self, samples: np.ndarray, prior: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the bandwidth based on Optuna v3.0 TPE.
        """
        if self._space_dim is None:
            raise ValueError("space_dim (the dimension of the space) must be provided for Optuna bandwidth.")

        domain_range = self._hard_ub - self._hard_lb
        means = np.append(samples, 0.5 * (self._hard_lb + self._hard_ub)) if prior else samples.copy()
        size = max(1, means.size)
        bandwidth = np.full_like(means, 0.2 * domain_range * size ** (- 1.0 / (self._space_dim + 4.0)))
        return means, bandwidth

    def _calculate(self, samples: np.ndarray, min_bandwidth_factor: float, prior: bool, compress: bool) -> None:
        """
        Calculate parameters of KDE based on Scott's rule

        Args:
            samples (np.ndarray): Samples to use for the construction of
                                  the parzen estimator

        NOTE:
            The bandwidth is computed using the following reference:
                * Scott, D.W. (1992) Multivariate Density Estimation:
                  Theory, Practice, and Visualization.
                * Berwin, A.T. (1993) Bandwidth Selection in Kernel
                  Density Estimation: A Review. (page 12)
                * Nils, B.H, (2013) Bandwidth selection for kernel
                  density estimation: a review of fully automatic selector
                * Wolfgang, H (2005) Nonparametric and Semiparametric Models
        """
        if self._heuristic is not None and self._heuristic not in self._HEURISTICS:
            raise ValueError(f"heuristic must be in {self._HEURISTICS}, but got {self._heuristic}")

        domain_range = self.ub - self.lb
        min_bandwidth = min_bandwidth_factor * domain_range
        if self._heuristic == HYPEROPT:
            means, bandwidth = self._bandwidth_heuristic_hyperopt(samples, prior)
        elif self._heuristic == OPTUNA:
            means, bandwidth = self._bandwidth_heuristic_optuna(samples, prior)
        else:
            preproc = self._preproc_with_compress if compress else self._preproc
            means, std, IQR = preproc(samples, prior)
            bandwidth = np.full_like(means, 1.059 * min(IQR / 1.34, std) * means.size ** (-0.2))

        # 99% of samples will be confined in mean \pm 0.025 * domain_range (2.5 sigma)
        clipped_bandwidth = np.clip(bandwidth, min_bandwidth, 0.5 * domain_range)
        if prior:
            clipped_bandwidth[-1] = domain_range  # The bandwidth for the prior

        self._means, self._stds = means, clipped_bandwidth
        self._norm_consts, self._logpdf_consts = calculate_norm_consts(
            lb=self.lb, ub=self.ub, means=self._means, stds=self._stds
        )

    @property
    def domain_size(self) -> NumericType:
        domain_range = self._hard_ub - self._hard_lb
        if self.q is None:
            return domain_range
        else:
            return int(np.round(domain_range / self.q)) + 1

    @property
    def size(self) -> int:
        return self._size

    @property
    def lb(self) -> NumericType:
        return self._lb

    @property
    def ub(self) -> NumericType:
        return self._ub

    @property
    def q(self) -> Optional[NumericType]:
        return self._q


class CategoricalParzenEstimator(AbstractParzenEstimator):
    def __init__(
        self,
        samples: np.ndarray,
        n_choices: int,
        top: float,
        *,
        prior: bool = True,
        weights: Optional[np.ndarray] = None,
    ):

        self._size = samples.size + prior
        self._weights = weights.copy() if weights is not None else uniform_weight(samples.size + prior)

        self._dtype = np.int32
        self._samples = np.append(samples, NULL_VALUE) if prior else samples.copy()
        self._n_choices = n_choices
        # AitchisonAitkenKernel: p = top or (1 - top) / (c - 1)
        # UniformKernel: p = 1 / c
        self._top, self._bottom, self._uniform = top, (1 - top) / (n_choices - 1), 1.0 / n_choices
        self._validate(samples, n_choices)

        self._probs = self._get_probs(samples, prior)
        bls = self._get_basislikelihoods(samples, prior)
        self._basis_loglikelihoods = np.log(bls)
        self._cum_basis_likelihoods = np.cumsum(bls, axis=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_choices={self.n_choices}, top={self._top}, probs={self._probs})"

    def _validate(self, samples: np.ndarray, n_choices: int) -> None:
        self._validate_weights()
        if samples.dtype not in [np.int32, np.int64]:
            raise ValueError(
                f"samples for {self.__class__.__name__} must be np.ndarray[np.int32/64], " f"but got {samples.dtype}."
            )
        if np.any(samples < 0) or np.any(samples >= n_choices):
            raise ValueError("All the samples must be in [0, n_choices).")

    def _get_basislikelihoods(self, samples: np.ndarray, prior: bool) -> np.ndarray:
        n_choices = self.n_choices
        likelihood_choices = np.array(
            [[self._top if i == j else self._bottom for j in range(n_choices)] for i in range(n_choices)]
        )

        # shape = (n_basis, n_choices)
        if prior:
            blls = np.vstack([likelihood_choices[samples], np.full(n_choices, self._uniform)])
        else:
            blls = likelihood_choices[samples]

        return np.maximum(1e-12, blls)

    def _get_probs(self, samples: np.ndarray, prior: bool) -> np.ndarray:
        n_choices = self.n_choices
        # if we use prior, apply uniform prior so that the initial value is 1 / c
        probs = np.full(n_choices, self._uniform * prior * self._weights[-1])

        weights = self._weights[:-1] if prior else self._weights
        masks = samples == np.arange(n_choices)[:, np.newaxis]
        slicer = np.arange(n_choices)
        for c, mask in enumerate(masks):
            weight = np.sum(weights[mask])
            probs[slicer != c] += weight * self._bottom
            probs[slicer == c] += weight * self._top

        if not np.isclose(np.sum(probs), 1.0):
            raise ValueError(f"Probabilities do not sum to 1, but got {probs}")
        else:
            probs /= np.sum(probs)

        return probs

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = x * (self.n_choices - 1)
        return scaled_x.astype(self._dtype)

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        return self._basis_loglikelihoods[:, x]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._probs[x]

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        return rng.choice(self.n_choices, p=self._probs, size=n_samples)

    def sample_by_indices(self, rng: np.random.RandomState, indices: np.ndarray) -> np.ndarray:
        n_samples = indices.size
        # equiv to ==> [rng.choice(n_choices, p=basis_likelihoods[idx], size=1)[0] for idx in indices]
        return (self._cum_basis_likelihoods[indices] > rng.random(n_samples)[:, np.newaxis]).argmax(axis=-1)

    @property
    def domain_size(self) -> NumericType:
        return self.n_choices

    @property
    def size(self) -> int:
        return self._size

    @property
    def n_choices(self) -> int:
        return self._n_choices


ParzenEstimatorType = Union[NumericalParzenEstimator, CategoricalParzenEstimator]


def _get_config_info(
    config: NumericalHPType, is_ordinal: bool
) -> Tuple[Optional[NumericType], bool, NumericType, NumericType]:
    if is_ordinal:
        info = config.meta
        q, log, lb, ub = info.get("q", None), info.get("log", False), info["lower"], info["upper"]
    else:
        q, log, lb, ub = config.q, config.log, config.lower, config.upper

    return q, log, lb, ub


def _convert_info_for_discrete(
    dtype: Type[Union[float, int]],
    q: Optional[NumericType],
    log: bool,
    lb: NumericType,
    ub: NumericType,
) -> Tuple[Optional[NumericType], Optional[NumericType], Optional[NumericType], NumericType, NumericType]:

    hard_lb: Optional[NumericType] = None
    hard_ub: Optional[NumericType] = None
    if dtype is int or q is not None:
        if log:
            q = None
        elif q is None:
            q = 1
        if q is not None:
            hard_lb, hard_ub = lb, ub
            lb -= 0.5 * q
            ub += 0.5 * q

    return q, hard_lb, hard_ub, lb, ub


def _convert_info_for_log_scale(
    vals: np.ndarray,
    dtype: Type[Union[float, int]],
    q: Optional[NumericType],
    lb: NumericType,
    ub: NumericType,
) -> Tuple[np.ndarray, NumericType, NumericType, Type[float]]:

    if q is not None:
        raise TypeError(f"q must be None for log scale, but got {q}")

    dtype = float
    lb, ub = np.log(lb), np.log(ub)
    vals = np.log(vals)
    return vals, lb, ub, dtype


def build_numerical_parzen_estimator(
    config: NumericalHPType,
    dtype: Type[Union[float, int]],
    vals: np.ndarray,
    is_ordinal: bool,
    *,
    default_min_bandwidth_factor: float = 1e-2,
    default_min_bandwidth_factor_for_discrete: Optional[float] = 1.0,
    magic_clip: bool = False,
    magic_clip_exponent: float = 1.0,
    prior: bool = True,
    weights: Optional[np.ndarray] = None,
    heuristic: Optional[Literal["hyperopt", "optuna"]] = None,
    space_dim: Optional[int] = None,
) -> NumericalParzenEstimator:
    """
    Build a numerical parzen estimator

    Args:
        config (NumericalHPType): Hyperparameter information from the ConfigSpace
        dtype (Type[np.number]): The data type of the hyperparameter
        vals (np.ndarray): The observed hyperparameter values
        is_ordinal (bool): Whether the configuration is ordinal
        weights (Optional[np.ndarray]): The weights for each basis.
        default_min_bandwidth_factor_for_discrete (float): .
        default_min_bandwidth_factor (float): .
        magic_clip (bool): Whether to apply the magic clipping in TPE.

    Returns:
        pe (NumericalParzenEstimator): Parzen estimator given a set of observations
    """
    min_bandwidth_factor = _get_min_bandwidth_factor(
        config=config,
        is_ordinal=is_ordinal,
        default_min_bandwidth_factor=default_min_bandwidth_factor,
        default_min_bandwidth_factor_for_discrete=default_min_bandwidth_factor_for_discrete,
    )
    q, log, lb, ub = _get_config_info(config=config, is_ordinal=is_ordinal)
    q, hard_lb, hard_ub, lb, ub = _convert_info_for_discrete(dtype=dtype, q=q, log=log, lb=lb, ub=ub)

    if log:
        vals, lb, ub, dtype = _convert_info_for_log_scale(vals=vals, dtype=dtype, q=q, lb=lb, ub=ub)

    pe = NumericalParzenEstimator(
        samples=vals,
        lb=lb,
        ub=ub,
        q=q,
        hard_lb=hard_lb,
        hard_ub=hard_ub,
        dtype=dtype,
        min_bandwidth_factor=min_bandwidth_factor,
        prior=prior,
        weights=weights,
        magic_clip=magic_clip,
        magic_clip_exponent=magic_clip_exponent,
        heuristic=heuristic,
        space_dim=space_dim,
    )

    return pe


def build_categorical_parzen_estimator(
    config: CategoricalHPType,
    vals: np.ndarray,
    top: float = 1.0,
    *,
    prior: bool = True,
    weights: Optional[np.ndarray] = None,
    vals_is_indices: bool = False,
) -> CategoricalParzenEstimator:
    """
    Build a categorical parzen estimator

    Args:
        config (CategoricalHPType): Hyperparameter information from the ConfigSpace
        vals (np.ndarray): The observed hyperparameter values (i.e. symbols, but not indices)
        top (float): The hyperparameter to define the probability of the category.
        weights (Optional[np.ndarray]): The weights for each basis.
        vals_is_indices (bool): Whether the vals is an array of indices or choices.

    Returns:
        pe (CategoricalParzenEstimator): Parzen estimators given a set of observations
    """
    choices = config.choices
    n_choices = len(choices)

    if vals_is_indices:
        choice_indices = vals
    else:
        try:
            choice2index = {choice: idx for idx, choice in enumerate(choices)}
            choice_indices = np.array([choice2index[val] for val in vals], dtype=np.int32)
        except KeyError:
            raise ValueError(
                "vals to build categorical parzen estimator must be "
                f"the list of symbols {choices}, but got the list of indices."
            )

    pe = CategoricalParzenEstimator(
        samples=choice_indices,
        n_choices=n_choices,
        top=top,
        prior=prior,
        weights=weights,
    )

    return pe
