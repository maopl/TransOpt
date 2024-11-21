import abc
import numpy as np
import copy
from typing import Union, Dict, List

from transopt.optimizer.acquisition_function.sequential import Sequential
from transopt.optimizer.optimizer_base.base import OptimizerBase
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.utils.serialization import (multioutput_to_ndarray,
                                          output_to_ndarray)


class Bilevel(OptimizerBase):
    """
    The abstract Model for Bilevel Optimization
    """
    def __init__(self, Refiner, Sampler, ACF, Pretrain, Model, Normalizer, config):
        super(Bilevel, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.upper_space = None
        self.lower_space = None
        self.mapping = None
        self.ini_num = None

        self.SpaceRefiner = None
        
        self.Sampler = Sampler
        self.ACF = ACF
        self.Pretrain = Pretrain
        
        self.LowerModel = Model
        self.UpperModel = Model
        
        self.Normalizer = Normalizer

        
        self.ACF.link_model(model=self.Model)

        self.MetaData = None
        
    
    def link_task(self, task_name:str, search_space: SearchSpace):
        self.task_name = task_name
        self.search_space = search_space
        
        self.upper_space = self.search_space.upper_space
        self.lower_space = self.search_space.lower_space
        
        
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.ACF.link_space(self.search_space)
        self.evaluator = Sequential(self.ACF, batch_size=1)
            
    
    def search_space_refine(self, metadata = None, metadata_info = None):
        pass
            
    def sample_initial_set(self, metadata = None, metadata_info = None):
        upper_samples = self.Sampler.sample(self.upper_space, self.ini_num)
        lower_samples = self.Sampler.sample(self.lower_space, self.ini_num)
        
        samples = np.hstack((upper_samples, lower_samples))
        return samples
    
    def pretrain(self, metadata = None, metadata_info = None):
        if self.Pretrain:
            self.Pretrain.set_data(metadata, metadata_info)
            self.Pretrain.meta_train()
    
    
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

        Y = copy.deepcopy(self._Y)
            
        X = copy.deepcopy(self._X)
        
        self.Model.fit(X, Y, optimize = True)
            
    def suggest(self):
        suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
        # suggested_sample = self.search_space.zip_inputs(suggested_sample)

        if self.Normalizer:
            suggested_sample = self.Normalizer.inverse_transform(X=suggested_sample)[0]
        
        return suggested_sample

        
    def observe(self, X: np.ndarray, Y: List[Dict]) -> None:
        # Check if the lists are empty and return if they are
        if X.shape[0] == 0 or len(Y) == 0:
            return

        Y = np.array(output_to_ndarray(Y))
        if self.Normalizer:
            self.Normalizer.fit(X, Y)
            X, Y = self.Normalizer.transform(X, Y)
        
        self._X = np.vstack((self._X, X)) if self._X.size else X
        self._Y = np.vstack((self._Y, Y)) if self._Y.size else Y
