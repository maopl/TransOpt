import abc
import numpy as np
import math
from typing import Union, Dict, List
from transopt.optimizer.acquisition_function.sequential import Sequential
from transopt.optimizer.optimizer_base.base import OptimizerBase
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.utils.serialization import (multioutput_to_ndarray,
                                          output_to_ndarray)



class EVOBase(OptimizerBase):
    """
    The abstract Model for Evolutionary Optimization
    """
    def __init__(self, config):
        super(EVOBase, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.search_space = None
        self.design_space = None
        self.mapping = None
        self.ini_num = None
        self.population = None
        self.pop_size = None
    
    def link_task(self, task_name:str, search_space: SearchSpace):
        self.task_name = task_name
        self.search_space = search_space
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))

