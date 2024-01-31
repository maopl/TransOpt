import pandas as pd
import numpy  as np
from abc import ABC, abstractmethod

class Parameter(ABC):
    def __init__(self, param_info):
        self.param_info = param_info
        self.name = param_info['name']
        self.type = param_info['type']

    @abstractmethod
    def sample(self, num = 1) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, x : np.array) -> np.array:
        pass

    @abstractmethod
    def inverse_transform(self, x : np.array) -> np.array:
        pass

    @property
    @abstractmethod
    def opt_lb(self) -> float:
        pass

    @property
    @abstractmethod
    def opt_ub(self) -> float:
        pass

    @property
    @abstractmethod
    def opt_default(self) -> float:
        pass
