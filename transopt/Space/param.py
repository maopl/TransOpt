import pandas as pd
import numpy  as np
from abc import ABC, abstractmethod

class Param(ABC):
    def __init__(self, param_info):
        self.param_info = param_info
        self.name       = param_info['name']
        pass

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
    def is_numeric(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """
        Integer and categorical variable
        """
        pass

    @property
    @abstractmethod
    def is_discrete_after_transform(self) -> bool:
        pass

    @property
    def is_categorical(self) -> bool:
        return not self.is_numeric


    @property
    @abstractmethod
    def opt_lb(self) -> float:
        pass

    @property
    @abstractmethod
    def opt_ub(self) -> float:
        pass