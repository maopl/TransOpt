import abc
import copy
import math
from typing import Dict, List, Union

import GPyOpt
import numpy as np

from transopt.optimizer.acquisition_function.sequential import Sequential
from transopt.optimizer.optimizer_base.base import OptimizerBase
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.utils.serialization import (multioutput_to_ndarray,
                                          output_to_ndarray)


class BO(OptimizerBase):
    """
    The abstract Model for Bayesian Optimization
    """

    def __init__(self, Refiner, Sampler, ACF, Pretrain, Model, DataSelector, Normalizer, config):
        super(BO, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.search_space = None
        self.ini_num = 10
        
        self.SpaceRefiner = Refiner
        self.Sampler = Sampler
        self.ACF = ACF
        self.Pretrain = Pretrain
        self.Model = Model
        self.DataSelector = DataSelector
        self.Normalizer = Normalizer

        
        self.ACF.link_model(model=self.Model)
        
        self.MetaData = None
    
    def link_task(self, task_name:str, search_sapce: SearchSpace):
        self.task_name = task_name
        self.search_space = search_sapce
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.ACF.link_space(self.search_space)
        self.evaluator = Sequential(self.ACF, batch_size=1)
            
    
    def search_space_refine(self, metadata = None):
        if self.SpaceRefiner is not None:
            self.search_space = self.SpaceRefiner.refine_space(self.search_space)
            self.ACF.link_space(self.search_space)
            self.evaluator = Sequential(self.ACF)
            
    def sample_initial_set(self, metadata = None):
        return self.Sampler.sample(self.search_space, self.ini_num)
    
    
    def meta_fit(self, metadata = None, metadata_info = None):
        if metadata:
            source_X = []
            source_Y = []
            for key, datasets in metadata.items():
                data_info = metadata_info[key]
                source_X.append(np.array([[data[var['name']] for var in data_info['variables']] for data in datasets]))
                source_Y.append(np.array([[data[var['name']] for var in data_info['objectives']] for data in datasets]))
                
            self.Model.meta_fit(source_X, source_Y)
    
    def fit(self):
        if self.Normalizer:
            Y = self.Normalizer.normalize(self._Y)
        else:
            Y = copy.deepcopy(self._Y)
            
        X = copy.deepcopy(self._X)
        
        if self.MetaData:
            pass
        elif self.DataSelector:
            pass
        else:
            pass
        self.Model.fit(X, Y, optimize = True)
            
    def suggest(self):
        suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
        # suggested_sample = self.search_space.zip_inputs(suggested_sample)

        return suggested_sample

        

    def observe(self, X: np.ndarray, Y: List[Dict]) -> None:

        # Check if the lists are empty and return if they are
        if X.shape[0] == 0 or len(Y) == 0:
            return

        self._X = np.vstack((self._X, X)) if self._X.size else X
        self._Y = np.vstack((self._Y, np.array(output_to_ndarray(Y)))) if self._Y.size else np.array(output_to_ndarray(Y))


