import numpy as np
import pandas as pd

from .param import Parameter

class ContinuousParameter(Parameter):
    def __init__(self, param_info):
        super().__init__(param_info)
        self.lb = param_info['range'][0]
        self.ub = param_info['range'][1]
        self.default = param_info.get('default', None)
        
    def sample(self, num=1) -> pd.DataFrame:
        assert num > 0, "Number of samples must be positive"
        samples = np.random.uniform(self.opt_lb, self.opt_ub, size=num)
        return pd.DataFrame(samples, columns=[self.name])

    def transform(self, x: np.array) -> np.array:
        return x

    def inverse_transform(self, x: np.array) -> np.array:
        return x

    @property
    def opt_lb(self) -> float:
        return self.lb

    @property
    def opt_ub(self) -> float:
        return self.ub
   
    @property
    def opt_default(self) -> float:
        return self.default 