import numpy as np
import pandas as pd

from transopt.utils.Register import para_register

from .param import BaseParameter


@para_register('continuous')
class ContinuousParameter(BaseParameter):
    def __init__(self, param_info):
        super().__init__(param_info)
        self.lb = param_info['lb']
        self.ub = param_info['ub']
        
    def sample(self, num=1):
        assert num > 0, "Number of samples must be positive"
        return np.random.uniform(self.opt_lb, self.opt_ub, size=num)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    @property
    def is_discrete_after_transform(self) -> bool:
        return False

    @property
    def opt_lb(self) -> float:
        return self.lb

    @property
    def opt_ub(self) -> float:
        return self.ub
