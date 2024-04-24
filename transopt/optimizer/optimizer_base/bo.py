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
        self.ini_num = None
        
        self.SpaceRefiner = Refiner
        self.Sampler = Sampler
        self.ACF = ACF
        self.Pretrain = Pretrain
        self.Model = Model
        self.DataSelector = DataSelector
        
        self.ACF.link_model(model=self.Model)
        
        self.MetaData = None
    
    def link_task(self, task_name:str, search_sapce: SearchSpace):
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
            self.search_space = self.SpaceRefiner.refine_space(space_info)
            self.acqusition = self.ACF.link_space(self.search_space)
            self.evaluator = Sequential(self.acqusition)
            
    
    
    def optimize(self, testsuits, data_handler):

        self.set_DataHandler(data_handler)
        while (testsuits.get_unsolved_num()):
            space_info = testsuits.get_cur_space_info()
            self.reset(testsuits.get_curname(), space_info, search_sapce=None)
            testsuits.sync_query_num(len(self._X))
            self.set_metadata()
            while (testsuits.get_rest_budget()):
                suggested_sample = self.suggest()
                observation = testsuits.f(suggested_sample)
                self.observe(suggested_sample, observation)
                if self.verbose:
                    self.visualization(testsuits, suggested_sample)
            testsuits.roll()
            
    def suggest():
        pass
        

    def observe(self, input_vectors: Union[List[Dict], Dict], output_value: Union[List[Dict], Dict]) -> None:

        self._data_handler.add_observation(input_vectors, output_value)

        # Convert dict to list of dict
        if isinstance(input_vectors, Dict):
            input_vectors = [input_vectors]
        if isinstance(output_value, Dict):
            output_value = [output_value]

        # Check if the lists are empty and return if they are
        if len(input_vectors) == 0 and len(output_value) == 0:
            return


        self._validate_observation('design', input_vectors=input_vectors, output_value=output_value)
        X = self.transform(input_vectors)

        self._X = np.vstack((self._X, vectors_to_ndarray(self._get_var_name('search'), X))) if self._X.size else vectors_to_ndarray(self._get_var_name('search'), X)
        if self.num_objective >= 2:
            self._Y = np.hstack((self._Y, multioutput_to_ndarray(output_value, self.num_objective))) if self._Y.size else multioutput_to_ndarray(output_value, self.num_objective)
        else:
            self._Y = np.vstack((self._Y, output_to_ndarray(output_value))) if self._Y.size else output_to_ndarray(output_value)


        
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


