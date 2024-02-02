import numpy as np
import pandas as pd

from transopt.utils.Register import para_register

from .param import BaseParameter


@para_register('integer')
class IntegerParameter(BaseParameter):
    def __init__(self, info):
        super().__init__(info)
        self.lb = info['lb']
        self.ub = info['ub']

    def sample(self, num_samples=1):
        assert num_samples > 0, "Number of samples must be positive"
        return np.random.randint(self.lb, self.ub + 1, size=num_samples)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        # Round the values to the nearest integer
        return np.round(x).astype(int)

    @property
    def is_discrete_after_transform(self) -> bool:
        return True
    
    @property
    def opt_lb(self):
        return self.lb

    @property
    def opt_ub(self):
        return self.ub