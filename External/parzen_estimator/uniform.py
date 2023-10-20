from typing import Optional, Type, Union

import numpy as np

from External.parzen_estimator.constants import (
    NumericType,
)
from External.parzen_estimator.constants import SQR2, SQR2PI
from External.parzen_estimator.parzen_estimator import AbstractParzenEstimator
from External.parzen_estimator.utils import calculate_norm_consts, erf, exp, validate_and_update_dtype, validate_and_update_q


class NumericalUniform(AbstractParzenEstimator):
    def __init__(
        self,
        lb: NumericType,
        ub: NumericType,
        *,
        q: Optional[NumericType] = None,
        dtype: Type[Union[np.number, int, float]] = np.float64,
    ):
        self._lb = lb
        self._ub = ub
        self._dtype = validate_and_update_dtype(dtype=dtype)
        self._q = validate_and_update_q(dtype=self._dtype, q=q)

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}(lb={self.lb}, ub={self.ub}, q={self.q})"
        return ret

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        if self.q is not None:
            val_range = int(self.domain_size / self.q + 0.5)
            samples = self.lb + self.q * rng.randint(val_range, size=n_samples)
        else:
            samples = rng.random(n_samples) * self.domain_size + self.lb

        return samples.astype(self._dtype)

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = self.lb + x * (self.ub - self.lb)
        scaled_x = scaled_x if self.q is None else np.round((scaled_x - self.lb) / self.q) * self.q + self.lb
        return scaled_x.astype(self._dtype)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.size, 1.0 / self.domain_size)

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} does not have basis.")

    @property
    def domain_size(self) -> NumericType:
        return self.ub - self.lb if self.q is None else int(np.round((self.ub - self.lb) / self.q)) + 1

    @property
    def lb(self) -> NumericType:
        return self._lb

    @property
    def ub(self) -> NumericType:
        return self._ub

    @property
    def q(self) -> Optional[NumericType]:
        return self._q

    @property
    def size(self) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} does not have size.")


class GaussianUniform(AbstractParzenEstimator):
    def __init__(
        self,
        lb: NumericType,
        ub: NumericType,
        *,
        q: Optional[NumericType] = None,
        dtype: Type[Union[np.number, int, float]] = np.float64,
    ):
        self._lb = lb
        self._ub = ub
        self._dtype = validate_and_update_dtype(dtype=dtype)
        self._q = validate_and_update_q(dtype=self._dtype, q=q)
        self._mean, self._std = (lb + ub) / 2, ub - lb
        norm_consts, logpdf_consts = calculate_norm_consts(
            lb, ub, means=np.asarray([self._mean]), stds=np.asarray([self._std])
        )
        self._norm_const, self._logpdf_const = norm_consts[0], logpdf_consts[0]

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}(lb={self.lb}, ub={self.ub}, q={self.q})"
        return ret

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        samples = np.zeros(n_samples, dtype=self._dtype)
        n_filled = 0
        while n_filled < n_samples:
            n_res = n_samples - n_filled
            vals = rng.normal(loc=self._mean, scale=self._std, size=n_res)
            valid_vals = vals[(self.lb <= vals) & (vals <= self.ub)]
            valid_vals = valid_vals if self.q is None else np.round((valid_vals - self.lb) / self.q) * self.q + self.lb
            n_valid = valid_vals.size
            samples[n_filled:n_filled+n_valid] = valid_vals
            n_filled += n_valid

        return samples

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = self.lb + x * (self.ub - self.lb)
        scaled_x = scaled_x if self.q is None else np.round((scaled_x - self.lb) / self.q) * self.q + self.lb
        return scaled_x.astype(self._dtype)

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
        z = (x - self._mean) / (SQR2 * self._std)
        return self._norm_const * 0.5 * (1.0 + erf(z))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.q is None:
            norm_const = self._norm_const / (SQR2PI * self._std)  # noqa: F841
            mahalanobis = ((x - self._mean) / self._std) ** 2  # noqa: F841
            return norm_const * exp(-0.5 * mahalanobis)
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x - 0.5 * self.q, self.lb))
            return integral_u - integral_l

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} does not have basis.")

    @property
    def domain_size(self) -> NumericType:
        return self.ub - self.lb if self.q is None else int(np.round((self.ub - self.lb) / self.q)) + 1

    @property
    def lb(self) -> NumericType:
        return self._lb

    @property
    def ub(self) -> NumericType:
        return self._ub

    @property
    def q(self) -> Optional[NumericType]:
        return self._q

    @property
    def size(self) -> int:
        return 1


class CategoricalUniform(AbstractParzenEstimator):
    def __init__(self, n_choices: int):
        self._n_choices = n_choices
        self._dtype = np.int32

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_choices={self.n_choices})"

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        return rng.randint(self.n_choices, size=n_samples)

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = x * (self.n_choices - 1)
        return scaled_x.astype(self._dtype)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.size, 1.0 / self.domain_size)

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} does not have basis.")

    @property
    def domain_size(self) -> NumericType:
        return self.n_choices

    @property
    def n_choices(self) -> int:
        return self._n_choices

    @property
    def size(self) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} does not have size.")
