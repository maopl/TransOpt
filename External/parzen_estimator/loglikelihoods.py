from typing import Optional

import numpy as np

from External.parzen_estimator.utils import logsumexp


def compute_config_loglikelihoods(basis_loglikelihoods: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """
    Calculate the loglikelihood of configurations based on parzen estimator.

    Args:
        basis_loglikelihoods (np.ndarray):
            The array of basis loglikelihood with the shape of (dim, n_basis, n_ei_candidates)

    Returns:
        config_loglikelihoods:
            The loglikelihoods of configs (n_ei_candidates, )
    """
    # Product of kernels with respect to dimension
    config_basis_loglikelihoods = basis_loglikelihoods.sum(axis=0)
    # Compute config loglikelihoods using logsumexp to avoid overflow
    config_loglikelihoods = logsumexp(config_basis_loglikelihoods, weights=weights, axis=0)

    return config_loglikelihoods
