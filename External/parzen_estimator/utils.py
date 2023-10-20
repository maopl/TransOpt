from typing import Optional, Tuple, Type, Union

import ConfigSpace as CS

import numpy as np

from External.parzen_estimator.constants import (
    NumericType,
    SQR2,
    SQR2PI,
    config2type,
)

from torch import as_tensor
from torch import erf as torch_erf
from torch import exp as torch_exp
from torch import log as torch_log
from torch import logsumexp as torch_logsumexp


# torch implementation is quicker than that of numpy!
def log(x: np.ndarray) -> np.ndarray:
    return torch_log(as_tensor(x)).cpu().detach().numpy()


def logsumexp(x: np.ndarray, axis: Optional[int], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Output log sum exp(x).

    NOTE:
        log sum w * exp(x) = log sum exp(log(w)) * exp(x)
                           = log sum exp(x + log(w))
    """
    if weights is not None:
        x += log(weights)[:, np.newaxis]

    return torch_logsumexp(as_tensor(x), axis=axis).cpu().detach().numpy()


def exp(x: np.ndarray) -> np.ndarray:
    return torch_exp(as_tensor(x)).cpu().detach().numpy()


def erf(x: np.ndarray) -> np.ndarray:
    return torch_erf(as_tensor(x)).cpu().detach().numpy()


def _get_min_bandwidth_factor(
    config: CS.hyperparameters,
    is_ordinal: bool,
    default_min_bandwidth_factor: float,
    default_min_bandwidth_factor_for_discrete: Optional[float],
) -> float:

    disc_exist = default_min_bandwidth_factor_for_discrete is not None
    if config.meta is not None and "min_bandwidth_factor" in config.meta:
        return config.meta["min_bandwidth_factor"]
    if is_ordinal and disc_exist:
        return default_min_bandwidth_factor_for_discrete / len(config.sequence)

    dtype = config2type[config.__class__.__name__]
    lb, ub, log, q = config.lower, config.upper, config.log, config.q

    if not log and (q is not None or dtype is int) and disc_exist:
        q = q if q is not None else 1
        n_grids = int((ub - lb) / q) + 1
        return default_min_bandwidth_factor_for_discrete / n_grids

    return default_min_bandwidth_factor


def calculate_norm_consts(
    lb: NumericType, ub: NumericType, means: np.ndarray, stds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        lb (NumericType):
            The lower bound of a parameter.
        ub (NumericType):
            The upper bound of a parameter.
        means (np.ndarray):
            The mean value for each kernel basis. The shape is (n_samples, ).
        stds (np.ndarray):
            The bandwidth value for each kernel basis. The shape is (n_samples, ).

    Returns:
        norm_consts, logpdf_consts (Tuple[np.ndarray, np.ndarray]):
            - norm_consts (np.ndarray):
                The normalization constants of each kernel due to the truncation.
            - logpdf_consts (np.ndarray):
                The constants for loglikelihood computation.
    """
    zl = (lb - means) / (SQR2 * stds)
    zu = (ub - means) / (SQR2 * stds)
    norm_consts = 2.0 / (erf(zu) - erf(zl))
    logpdf_consts = log(norm_consts / (SQR2PI * stds))
    return norm_consts, logpdf_consts


def validate_and_update_dtype(dtype: Type[Union[np.number, int, float]]) -> Type[np.number]:
    dtype_choices = (np.int32, np.int64, np.float32, np.float64)
    if dtype is int:
        return np.int32
    elif dtype is float:
        return np.float64
    elif dtype in dtype_choices:
        return dtype  # type: ignore
    else:
        raise ValueError(f"dtype for numerical pdf must be {dtype_choices}, but got {dtype}")


def validate_and_update_q(dtype: Type[np.number], q: Optional[NumericType]) -> Optional[NumericType]:
    if dtype not in (np.int32, np.int64):
        return q

    if not isinstance(q, int):
        raise TypeError(f"q must be integer if dtype is np.int32/64, but got {q}")

    return q if q is not None else 1
