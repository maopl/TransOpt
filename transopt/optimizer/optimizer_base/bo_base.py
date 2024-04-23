import abc
import math
from typing import Dict, List, Union

import ConfigSpace
import GPyOpt
import numpy as np

from optimizer.acquisition_function.get_acf import get_acf
from sampler.get_sampler import get_sampler


from optimizer.acquisition_function.sequential import Sequential
from optimizer.optimizer_base.base import OptimizerBase

from transopt.utils.Visualization import visual_pf


class BOBase(OptimizerBase):
    """
    The abstract Model for Bayesian Optimization
    """

    def __init__(self, config):
        super(BOBase, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.search_space = None
        self.design_space = None
        self.ini_num = None
        
        assert 'refiner' in self.config
        SpaceRefiner = get_refiner(self.config['refiner'])
        
        assert 'sampler' in self.config
        Sampler = get_sampler(self.config['sampler'])
        
    
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
        
        
        # self.model_list = []
        # self.kernel_list = []

    # def _get_var_bound(self, space_name)->Dict:
    #     assert self.design_space is not None
    #     assert self.search_space is not None
    #     if space_name == 'design':
    #         return  self.design_bounds
    #     elif space_name == 'search':
    #         return self.search_bounds
    #     else:
    #         raise NameError('Wrong space name, choose space name from design or search!')


    # def _get_var_type(self, space_name)->Dict:
    #     assert self.design_space is not None
    #     assert self.search_space is not None
    #     var_type = {}
    #     if space_name == 'design':
    #         return self.design_type
    #     elif space_name == 'search':
    #         return self.search_type
    #     else:
    #         raise NameError('Wrong space name, choose space name from design or search!')


    # def _get_var_name(self, space_name)->List:
    #     assert self.design_space is not None
    #     assert self.search_space is not None
    #     name = []
    #     if space_name == 'design':
    #         space = self.design_space.config_space
    #     elif space_name == 'search':
    #         space = self.search_space.config_space
    #     else:
    #         raise NameError('Wrong space name, choose space name from design or search!')

    #     name = [i['name'] for i in space]

    #     return name

    # def _validate_observation(self, space_name: str, input_vectors: Union[List[Dict], Dict],
    #                           output_value: Union[List[Dict], Dict]) -> None:
    #     # If we have single dictionary, convert it to list
    #     if isinstance(input_vectors, dict):
    #         input_vectors = [input_vectors]
    #     if isinstance(output_value, dict):
    #         output_value = [output_value]

    #     # Get the variable bounds and types for the specified space
    #     var_bounds = self._get_var_bound(space_name)

    #     # Validate input_vectors
    #     for iv in input_vectors:
    #         for var_name, value in iv.items():
    #             if var_name not in var_bounds:
    #                 raise ValueError(f"Variable {var_name} is not in the specified {space_name} space.")
    #             if not var_bounds[var_name][0] <= value <= var_bounds[var_name][1]:
    #                 raise ValueError(
    #                     f"Value for {var_name} is out of bounds. Expected value between {var_bounds[var_name][0]} and {var_bounds[var_name][1]} but got {value}.")
    #             # You can further validate based on var_types if there are specific constraints based on type

    #     # Validate output_value
    #     for ov in output_value:
    #         self._validate_output_value(ov)  # Assuming _validate_output_value is in the same class

    #     if len(input_vectors) != len(output_value):
    #         raise ValueError("Number of input vectors and output values must be the same.")



    # def _validate_output_value(self, output: Dict) -> None:
    #     """
    #     Validates the structure of the output value.

    #     Args:
    #     - output (Dict): A dictionary containing the output value and associated info.

    #     Raises:
    #     - ValueError: If the output value structure is not as expected.
    #     """
    #     if not isinstance(output, dict):
    #         raise ValueError("Expected output to be a dictionary.")




    def bind_task(self, space_info: Dict[str, dict]) -> List[dict]:
        """
        Define the space based on the provided space_info.

        :param space_info: Dictionary containing information about the variables, should include 'input_dim'.
        :return: Returns a list defining the space.
        """
        # Ensure 'input_dim' is present.
        space = []
        for key, var in space_info['variables'].items():
            var_dic = {
                'name': key,
                'domain': tuple(var['bounds'])
            }
            if var['type'] == 'UniformFloatHyperparameter':
                var_dic['type'] = 'continuous'
            elif var['type'] == 'UniformIntegerHyperparameter':
                var_dic['type'] = 'continuous'
            elif var['type'] == 'CategoricalHyperparameter':
                var_dic['type'] = 'discrete'
            else:
                raise NameError('Unknown variable type!')
            space.append(var_dic.copy())

        return space

    # def _set_default_search_space(self):
    #     assert self.design_space is not None
    #     space = []
    #     for var in self.design_space.config_space:
    #         var_dic = {
    #             'name': var['name'],
    #             'type': 'continuous',
    #             'domain': tuple([-1, 1])
    #         }

    #         space.append(var_dic.copy())

    #     return space


    # def validate_space_info(self, space_info: Dict) -> bool:
    #     """
    #     Validate the structure and content of space_info.

    #     :param space_info: Dictionary containing information about the space.
    #     :return: True if space_info is valid, False otherwise.
    #     """

    #     # Check if 'input_dim' is present.
    #     if 'input_dim' not in space_info:
    #         raise ValueError("'input_dim' must be present in space_info.")

    #     if 'budget' not in space_info:
    #         raise ValueError("'budget' must be present in space_info.")

    #     if 'variables' not in space_info:
    #         raise ValueError("'variables' must be present in space_info.")

    #     if 'num_objective' not in space_info:
    #         raise ValueError("'num_objective' must be present in space_info.")

    #     # Ensure the rest of the keys equal the count specified by 'input_dim'.
    #     input_dim = space_info['input_dim']
    #     if len(space_info['variables']) != input_dim:
    #         raise ValueError(f"Expected {input_dim} variable(s), but got {len(space_info['variables'])}.")

    #     # Validate each variable.
    #     for key, value in space_info['variables'].items():
    #         # Check if the variable's value is a dictionary.
    #         if not isinstance(value, dict):
    #             raise TypeError(f"Expected a dictionary for variable '{key}', but got {type(value).__name__}.")

    #         # Check if 'bounds' and 'type' are present in the dictionary.
    #         if 'bounds' not in value:
    #             raise KeyError(f"'bounds' is missing for variable '{key}'.")
    #         if 'type' not in value:
    #             raise KeyError(f"'type' is missing for variable '{key}'.")

    #     return True


    # def set_space(self, design_space_info: Dict[str, dict], search_space_info: Union[Dict[str, dict], None]= None):
    #     """
    #     Define the design space based on the provided space_info and copy the design space to the search space.

    #     :param design_space_info: Dictionary containing information about the variables.
    #     """
    #     assert self.validate_space_info(design_space_info)

    #     self.design_info = design_space_info.copy()
    #     self.input_dim = design_space_info['input_dim']
    #     self.num_objective = design_space_info['num_objective']
    #     self.budget = design_space_info['budget']

    #     if self.ini_num is None:
    #         self.ini_num = 11 * self.input_dim - 1

    #     task_design_space = self.bind_task(design_space_info)
    #     self.design_space = GPyOpt.Design_space(space=task_design_space)
    #     if search_space_info is not None:
    #         task_search_space = self.bind_task(search_space_info)
    #     else:
    #         task_search_space = self._set_default_search_space()
    #     self.search_space = GPyOpt.Design_space(space=task_search_space)

    #     self.design_params = self.design_info['variables']
    #     self.search_param = {}
    #     self.search_bounds = {k:[-1,1] for k, _ in self.design_params.items()}
    #     self.design_bounds = {}
    #     for k, v in self.design_params.items():
    #         if 'CategoricalHyperparameter' == self.design_params[k]['type']:
    #             db = [0, len(self.design_params[k]['bounds']) - 1]
    #             self.design_bounds[k] = db
    #         else:
    #             self.design_bounds[k] = v['bounds']

    #     self.search_type = {k:'continuous' for k, _ in self.design_params.items()}
    #     self.design_type = {}
    #     for k, v in self.design_params.items():
    #         self.search_param[k] = {'bounds': [-1,1], 'type': 'continuous'}
    #         if self.design_params[k]['type'] == 'UniformFloatHyperparameter':
    #             self.design_params[k]['type'] = 'continuous'
    #             self.design_type[k] = 'continuous'
    #         elif self.design_params[k]['type'] == 'UniformIntegerHyperparameter':
    #             self.design_params[k]['type'] = 'integer'
    #             self.design_type[k] = 'integer'
    #         elif self.design_params[k]['type'] == 'CategoricalHyperparameter':
    #             self.design_params[k]['type'] = 'categorical'
    #             self.design_type[k] = 'categorical'
    #         else:
    #             raise NameError('Unknown variable type!')


    # def get_spaceinfo(self, space_name):
    #     assert self.design_space is not None
    #     assert self.search_space is not None
    #     space_info = {}
    #     space_info['input_dim'] = self.input_dim
    #     space_info['num_objective'] = self.num_objective
    #     space_info['budget'] = self.budget
    #     space_info['variables'] = {}
    #     if space_name == 'design':
    #         for k, v in self.design_params.items():
    #             space_info['variables'][k] = {'bounds':v['bounds'], 'type':v['type']}
    #     elif space_name == 'search':
    #         for  k, v  in self.search_params.items():
    #             space_info['variables'][k] = {'bounds':v['bounds'], 'type':v['type']}
    #     else:
    #         raise NameError('Wrong space name, choose space name from design or search!')

    #     return space_info



    # def _to_searchspace(self, X: Union[ConfigSpace.Configuration, Dict]) -> Dict:
    #     xx = []
    #     search_bound_dic = self._get_var_bound('search')
    #     search_bounds =[]
    #     design_bound_dic = self._get_var_bound('design')
    #     design_bounds = []
    #     search_type_dic = self._get_var_type('search')
    #     search_type = []

    #     for  k, v in X.items():
    #         xx.append(v)
    #         search_bounds.append(search_bound_dic[k])
    #         design_bounds.append(design_bound_dic[k])
    #         search_type.append(search_type_dic[k])

    #     xx = np.array(xx)
    #     search_bounds = np.array(search_bounds)
    #     design_bounds = np.array(design_bounds)
    #     xx = (xx - design_bounds[:, 0]) * (search_bounds[:, 1] - search_bounds[:, 0]) / (design_bounds[:, 1] - design_bounds[:, 0]) + (search_bounds[:, 0])

    #     int_flag = [idx for idx, i in enumerate(search_type) if i == 'integer' or i == 'categorical']

    #     configuration_t = {k: np.round(xx[idx]).astype(int) if idx in int_flag else xx[idx] for idx, k in
    #                        enumerate(design_bound_dic.keys())}

    #     return configuration_t

    # def _to_designspace(self, X: Union[ConfigSpace.Configuration, Dict]) -> Dict:
    #     xx = []
    #     search_bound_dic = self._get_var_bound('search')
    #     search_bounds =[]
    #     design_bound_dic = self._get_var_bound('design')

    #     design_bounds = []
    #     design_type_dic = self._get_var_type('design')
    #     design_type = []
    #     for  k, v in X.items():
    #         xx.append(v)
    #         search_bounds.append(search_bound_dic[k])
    #         design_bounds.append(design_bound_dic[k])
    #         design_type.append(design_type_dic[k])

    #     xx = np.array(xx)
    #     search_bounds = np.array(search_bounds)
    #     design_bounds = np.array(design_bounds)

    #     xx = (xx - search_bounds[:, 0]) * (design_bounds[:, 1] - design_bounds[:, 0]) / (search_bounds[:, 1] - search_bounds[:, 0]) + (design_bounds[:, 0])

    #     int_flag = [idx for idx, i  in enumerate(design_type) if i == 'integer' or i == 'categorical']


    #     configuration_t = {k: np.round(xx[idx]).astype(int) if idx in int_flag else xx[idx] for idx, k in
    #                        enumerate(design_bound_dic.keys())}

    #     return configuration_t

    # def transform(self, X: Union[ConfigSpace.Configuration, Dict, List[Union[ConfigSpace.Configuration, Dict]]]) -> Union[Dict, List[Dict]]:
    #     """
    #     Transform configurations from design_space to search_space.

    #     :param X: Configurations in design_space.
    #     :return: Transformed configurations in search_space.
    #     """

    #     if isinstance(X, list):
    #         return [self._to_searchspace(x) for x in X]
    #     else:
    #         return self._to_searchspace(X)

    # def inverse_transform(self, X: Union[ConfigSpace.Configuration, Dict, List[Union[ConfigSpace.Configuration, Dict]]]) -> Union[Dict, List[Dict]]:
    #     """
    #     Transform configurations from search_space back to design_space.

    #     :param X: Configurations in search_space.
    #     :return: Transformed configurations back in design_space.
    #     """
    #     if isinstance(X, list):
    #         return [self._to_designspace(x) for x in X]
    #     else:
    #         return self._to_designspace(X)

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


        
    def set_DataHandler(self, data_handler:OptTaskDataHandler):
        self._data_handler = data_handler

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

    def reset(self, task_name:str, design_space:Dict, search_sapce:Union[None, Dict] = None):
        self.set_space(design_space, search_sapce)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self._data_handler.reset_task(task_name, design_space)
        self.sync_data(self._data_handler.get_input_vectors(), self._data_handler.get_output_value())
        self.model_reset()
        self.acqusition = get_ACF(self.acf, model=self, search_space=self.search_space, config=self.config)
        self.evaluator = Sequential(self.acqusition)

    # def visualization(self, testsuits, candidate):
    #     assert self.input_dim
    #     try:
    #         assert self.obj_model
    #     except:
    #         return

    #     try:
    #         assert self.acqusition
    #     except:
    #         self.acqusition = None

    #     if self.num_objective == 2:
    #         visual_pf(optimizer=self, train_x=self._X, train_y=self._Y,
    #                     testsuites=testsuits, ac_model=self.acqusition, Ac_candi=candidate)



    @abc.abstractmethod
    def model_reset(self):
        return

    @abc.abstractmethod
    def update_model(self, data: Dict):
        "Augment the dataset of the model"
        return

    @abc.abstractmethod
    def predict(self, X):
        "Get the predicted mean and std at X."
        return

    @abc.abstractmethod
    def get_fmin(self):
        "Get the minimum of the current model."
        return

    @abc.abstractmethod
    def initial_sample(self):
        return