import numpy as np
import pandas as pd

from transopt.utils.Register import para_register

from .param import BaseParameter


@para_register('int_exponent')
class IntExponentParameter(BaseParameter):
    """
    Integer parameter, search in log-scale, and exponent is integer.
    For example [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    """
    def __init__(self, info):
        super().__init__(info)
        self.base = info.get('base', 2)
        self.lb = info['lb']  # Lower bound of the exponent
        self.ub = info['ub']  # Upper bound of the exponent

    def sample(self, num_samples=1):
        assert num_samples > 0, "Number of samples must be positive"
        # Sample uniformly in the exponent space
        exponents = np.random.randint(self.opt_lb, self.opt_ub + 1, size=num_samples)
        return np.power(self.base, exponents).astype(int)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        # Transform to log-scale with the specified base
        return np.log(x) / np.log(self.base)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        # Inverse transform from log-scale and round to the nearest exponent
        return np.power(self.base, np.round(x)).astype(int)

    @property
    def is_discrete_after_transform(self) -> bool:
        return True
    
    @property
    def opt_lb(self) -> float:
        return np.log(self.lb) / np.log(self.base)

    @property
    def opt_ub(self) -> float:
        return np.log(self.ub) / np.log(self.base)
