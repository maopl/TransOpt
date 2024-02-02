import numpy as np
import pandas as pd

from transopt.utils.Register import para_register

from .param import BaseParameter

@para_register('log_continuous')
class LogContinuousParameter(BaseParameter):
    def __init__(self, info):
        super().__init__(info)
        self.lb = info['lb']  # Original lower bound of the parameter
        self.ub = info['ub']  # Original upper bound of the parameter
        self.base = info.get('base', np.e)  # Base of the logarithm, default is natural log

    def sample(self, num_samples=1):
        assert num_samples > 0, "Number of samples must be positive"
        # Sample uniformly in the log-scale space
        log_lb = np.log(self.lb) / np.log(self.base)
        log_ub = np.log(self.ub) / np.log(self.base)
        return np.power(self.base, np.random.uniform(log_lb, log_ub, size=num_samples))

    def transform(self, x: np.ndarray) -> np.ndarray:
        # Transform to log-scale with the specified base
        return np.log(x) / np.log(self.base)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        # Inverse transform from log-scale
        return np.power(self.base, x)

    @property
    def is_discrete_after_transform(self) -> bool:
        return False

    @property
    def opt_lb(self) -> float:
        return np.log(self.lb) / np.log(self.base)

    @property
    def opt_ub(self) -> float:
        return np.log(self.ub) / np.log(self.base)