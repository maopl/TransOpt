import abc
import math
from typing import Dict, List, Union

import ConfigSpace
import GPyOpt
import numpy as np

from transopt.optimizer.acquisition_function.sequential import Sequential
from transopt.optimizer.optimizer_base.base import OptimizerBase
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace


class BO(OptimizerBase):
    """
    The abstract Model for Bayesian Optimization
    """

    def __init__(self, Refiner, Sampler, ACF, Pretrain, Model, DataSelector, config):
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
        
        self.ACF.link_model(model=self.Model)
        
        self.MetaData = None
    
    def link_task(self, task_name:str, search_sapce: SearchSpace):
        self.task_name = task_name
        self.search_space = search_sapce
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.acqusition = self.ACF.link_space(self.search_space)
        self.evaluator = Sequential(self.acqusition)


    def set_metadata(self):
        if 'metadata' in self.config:
            pass
            
    
    def search_space_refine(self):
        if self.SpaceRefiner is not None:
            self.search_space = self.SpaceRefiner.refine_space(self.search_space)
            self.acqusition = self.ACF.link_space(self.search_space)
            self.evaluator = Sequential(self.acqusition)
            
    def sample_initial_set(self):
        return self.Sampler.sample(self.search_space, self.ini_num)
    
    
    def meta_fit(self):
        pass
            
    def suggest(self):
        if 'normalize' in self.config:
            self.normalizer = get_normalizer(self.config['normalize'])

        if self.aux_data is not None:
            pass
        else:
            self.aux_data = {}

        Data = self.combine_data()
        self.update_model(Data)
        suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
        suggested_sample = self.search_space.zip_inputs(suggested_sample)
        suggested_sample = ndarray_to_vectors(self._get_var_name('search'),suggested_sample)
        design_suggested_sample = self.inverse_transform(suggested_sample)

        return design_suggested_sample

        

    def observe(self, X: Union[List[Dict], Dict], Y: Union[List[Dict], Dict]) -> None:

        self._data_handler.add_observation(X, Y)

        # Convert dict to list of dict
        if isinstance(X, Dict):
            X = [X]
        if isinstance(Y, Dict):
            Y = [Y]

        # Check if the lists are empty and return if they are
        if len(X) == 0 and len(Y) == 0:
            return

        self._X = np.vstack((self._X, vectors_to_ndarray(self._get_var_name('search'), X))) if self._X.size else vectors_to_ndarray(self._get_var_name('search'), X)
        if self.num_objective >= 2:
            self._Y = np.hstack((self._Y, multioutput_to_ndarray(Y, self.num_objective))) if self._Y.size else multioutput_to_ndarray(Y, self.num_objective)
        else:
            self._Y = np.vstack((self._Y, output_to_ndarray(Y))) if self._Y.size else output_to_ndarray(Y)


        
    # def set_DataHandler(self, data_handler:OptTaskDataHandler):
    #     self._data_handler = data_handler

    def sync_data(self, input_vectors: List[Dict], output_value: List[Dict]):

        # Convert dict to list of dict
        if isinstance(input_vectors, Dict):
            input_vectors = [input_vectors]
        if isinstance(output_value, Dict):
            output_value = [output_value]
        self.get_spaceinfo('design')
        # Check if the lists are empty and return if they are
        if len(input_vectors) == 0 and len(output_value) == 0:
            return

        self._validate_observation('design', input_vectors=input_vectors, output_value=output_value)
        X = self.transform(input_vectors)

        self._X = np.vstack((self._X, vectors_to_ndarray(self._get_var_name('search'), X))) if self._X.size else vectors_to_ndarray(self._get_var_name('search'), X)
        if self.get_spaceinfo('design')['num_objective'] >= 2:
            self._Y = []
        else:
            self._Y = np.vstack((self._Y, output_to_ndarray(output_value))) if self._Y.size else output_to_ndarray(output_value)


