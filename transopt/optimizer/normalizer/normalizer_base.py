from abc import abstractmethod, ABC
from typing import Dict, Hashable
import numpy as np

class NormalizerBase(ABC):
    
    @abstractmethod
    def fit(self, X, Y):
        raise NotImplementedError
            
    def transform(self, X = None, Y = None):
        raise NotImplementedError

    def inverse_transform(self, X = None, Y = None):

        raise NotImplementedError 
    