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

    def __init__(self, Refiner, Sampler, ACF, Pretrain, Model, Selector, config):
        super(BO, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.search_space = None
        self.ini_num = None
        
        self.SpaceRefiner = Refiner
        self.Sampler = Sampler
        self.ACF_class = ACF
        self.Pretrain = Pretrain
        self.Model = Model
        self.Selector = Selector
    
    def bind(self, task_name:str, search_sapce: SearchSpace):
        self.search_space = search_sapce
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.Model.bind(task_name, search_sapce)
        self.acqusition = self.ACF.bind(model=self.Model, search_space=self.search_space, config=self.config)
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

    def set_metadata(self):
        if self._data_handler is None:
            self.aux_data = {}
        else:
            self.aux_data = self._data_handler.get_auxillary_data()




    # @abc.abstractmethod
    # def model_reset(self):
    #     return

    # @abc.abstractmethod
    # def update_model(self, data: Dict):
    #     "Augment the dataset of the model"
    #     return

    # @abc.abstractmethod
    # def predict(self, X):
    #     "Get the predicted mean and std at X."
    #     return

    # @abc.abstractmethod
    # def get_fmin(self):
    #     "Get the minimum of the current model."
    #     return

    # @abc.abstractmethod
    # def initial_sample(self):
    #     return