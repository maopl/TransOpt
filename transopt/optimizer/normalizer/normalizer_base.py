from abc import abstractmethod, ABC
from typing import Dict, Hashable
import numpy as np

class NormalizerBase(ABC):
    def __init__(self, config):
        self.config = config
    @abstractmethod
    def update(self, Y):
        raise NotImplementedError
    @abstractmethod 
    def transform(self, Y = None):
        raise NotImplementedError
    @abstractmethod
    def inverse_transform(self, Y = None):

        raise NotImplementedError 
    