import pandas as pd
import numpy  as np


class BaseParameter:
    def __init__(self, param_info):
        """ Base class for parameters in the design space. """
        self.param_info = param_info
        self.name = param_info['name']
        self.type = param_info['type']

    def sample(self, num = 1) -> pd.DataFrame:
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def inverse_transform(self, x):
        raise NotImplementedError

    @property
    def is_discrete_after_transform(self) -> bool:
        """ If the parameter is discrete in the optimization space. """
        raise NotImplementedError

    @property
    def opt_lb(self) -> float:
        raise NotImplementedError

    @property
    def opt_ub(self) -> float:
        raise NotImplementedError
