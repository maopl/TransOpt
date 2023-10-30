import abc
import numpy as np
import ConfigSpace
import math
from importlib_metadata import version
from typing import Union, Dict, List
from Optimizer.OptimizerBase import OptimizerBase
import GPyOpt
from Util.Data import InputData, TaskData, vectors_to_ndarray, output_to_ndarray
from KnowledgeBase.TaskDataHandler import OptTaskDataHandler


class BayesianOptimizerBase(OptimizerBase):
    """
    The abstract Model for Bayesian Optimization
    """

    def __init__(self, config):
        super(BayesianOptimizerBase, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.search_space = None
        self.design_space = None
        self.ini_num = None

    def _get_var_bound(self, space_name)->Dict:
        assert self.design_space is not None
        assert self.search_space is not None
        bound = {}
        if space_name == 'design':
            space = self.design_space.config_space
        elif space_name == 'search':
            space = self.search_space.config_space
        else:
            raise NameError('Wrong space name, choose space name from design or search!')

        for var in space:
            bound[var['name']] = var['domain']

        return bound

    def _get_var_type(self, space_name)->Dict:
        assert self.design_space is not None
        assert self.search_space is not None
        var_type = {}
        if space_name == 'design':
            space = self.design_space.config_space
        elif space_name == 'search':
            space = self.search_space.config_space
        else:
            raise NameError('Wrong space name, choose space name from design or search!')

        for var in space:
            var_type[var['name']] = var['type']

        return var_type

    def _get_var_name(self, space_name)->List:
        assert self.design_space is not None
        assert self.search_space is not None
        name = []
        if space_name == 'design':
            space = self.design_space.config_space
        elif space_name == 'search':
            space = self.search_space.config_space
        else:
            raise NameError('Wrong space name, choose space name from design or search!')

        name = [i['name'] for i in space]

        return name

    def _validate_observation(self, space_name: str, input_vectors: Union[List[Dict], Dict],
                              output_value: Union[List[Dict], Dict]) -> None:
        # If we have single dictionary, convert it to list
        if isinstance(input_vectors, dict):
            input_vectors = [input_vectors]
        if isinstance(output_value, dict):
            output_value = [output_value]

        # Get the variable bounds and types for the specified space
        var_bounds = self._get_var_bound(space_name)
        var_types = self._get_var_type(space_name)

        # Validate input_vectors
        for iv in input_vectors:
            for var_name, value in iv.items():
                if var_name not in var_bounds:
                    raise ValueError(f"Variable {var_name} is not in the specified {space_name} space.")
                if not var_bounds[var_name][0] <= value <= var_bounds[var_name][1]:
                    raise ValueError(
                        f"Value for {var_name} is out of bounds. Expected value between {var_bounds[var_name][0]} and {var_bounds[var_name][1]} but got {value}.")
                # You can further validate based on var_types if there are specific constraints based on type

        # Validate output_value
        for ov in output_value:
            self._validate_output_value(ov)  # Assuming _validate_output_value is in the same class

        if len(input_vectors) != len(output_value):
            raise ValueError("Number of input vectors and output values must be the same.")



    def _validate_output_value(self, output: Dict) -> None:
        """
        Validates the structure of the output value.

        Args:
        - output (Dict): A dictionary containing the output value and associated info.

        Raises:
        - ValueError: If the output value structure is not as expected.
        """
        if not isinstance(output, dict):
            raise ValueError("Expected output to be a dictionary.")

        if 'function_value' not in output:
            raise ValueError("Output value must contain 'function_value' key.")

        function_value = output['function_value']
        if not isinstance(function_value, (int, float)):
            raise ValueError("'function_value' should be an int or float.")

        # Check for NaN and infinite values
        if math.isnan(function_value) or math.isinf(function_value):
            raise ValueError("'function_value' should not be NaN or infinite.")

        if 'info' not in output:
            raise ValueError("Output value must contain 'info' key.")

        if not isinstance(output['info'], dict):
            raise ValueError("'info' key should contain a dictionary.")

        if 'fidelity' not in output['info']:
            raise ValueError("'info' dictionary must contain 'fidelity' key.")


    def _set_space(self, space_info: Dict[str, dict]) -> List[dict]:
        """
        Define the space based on the provided space_info.

        :param space_info: Dictionary containing information about the variables, should include 'input_dim'.
        :return: Returns a list defining the space.
        """
        # Ensure 'input_dim' is present.
        space = []
        for key, var in space_info.items():
            if key == 'input_dim' or key == 'budget' or key == 'seed' or key == 'task_id':
                continue

            var_dic = {
                'name': key,
                'domain': tuple(var['bounds'])
            }
            if var['type'] == 'UniformFloatHyperparameter':
                var_dic['type'] = 'continuous'
            elif var['type'] == 'UniformIntegerHyperparameter':
                var_dic['type'] = 'discrete'
            else:
                raise NameError('Unknown variable type!')
            space.append(var_dic.copy())

        return space

    def _set_default_search_space(self):
        assert self.design_space is not None
        space = []
        for var in self.design_space.config_space:
            var_dic = {
                'name': var['name'],
                'type': 'continuous',
                'domain': tuple([-1, 1])
            }

            space.append(var_dic.copy())

        return space

    def sync_from_handler(self, data_handler):
        self.observe(data_handler.get_input_vectors(), data_handler.get_output_value())
        self.set_auxillary_data(data_handler.get_auxillary_data())

    def set_auxillary_data(self, aux_data:Union[Dict, List[Dict], None]):
        if aux_data is None:
            self.aux_data = {}
        else:
            self.aux_data = aux_data

    def validate_space_info(self, space_info: Dict) -> bool:
        """
        Validate the structure and content of space_info.

        :param space_info: Dictionary containing information about the space.
        :return: True if space_info is valid, False otherwise.
        """

        # Check if 'input_dim' is present.
        if 'input_dim' not in space_info:
            raise ValueError("'input_dim' must be present in space_info.")

        if 'budget' not in space_info:
            raise ValueError("'budget' must be present in space_info.")

        if 'seed' not in space_info:
            raise ValueError("'seed' must be present in space_info.")

        if 'task_id' not in space_info:
            raise ValueError("'task_id' must be present in space_info.")

        # Ensure the rest of the keys equal the count specified by 'input_dim'.
        input_dim = space_info['input_dim']
        if len(space_info) - 4 != input_dim:
            raise ValueError(f"Expected {input_dim} variable(s), but got {len(space_info) - 4}.")

        # Validate each variable.
        for key, value in space_info.items():
            if key == 'input_dim' or 'budget':
                continue  # We've already checked 'input_dim'.

            # Check if the variable's value is a dictionary.
            if not isinstance(value, dict):
                raise TypeError(f"Expected a dictionary for variable '{key}', but got {type(value).__name__}.")

            # Check if 'bounds' and 'type' are present in the dictionary.
            if 'bounds' not in value:
                raise KeyError(f"'bounds' is missing for variable '{key}'.")
            if 'type' not in value:
                raise KeyError(f"'type' is missing for variable '{key}'.")


        return True


    def set_space(self, design_space_info: Dict[str, dict], search_space_info: Union[Dict[str, dict], None]= None):
        """
        Define the design space based on the provided space_info and copy the design space to the search space.

        :param design_space_info: Dictionary containing information about the variables.
        """
        assert self.validate_space_info(design_space_info)

        self.design_info = design_space_info.copy()
        self.input_dim = design_space_info['input_dim']
        self.budget = design_space_info['budget']

        if self.ini_num is None:
            self.ini_num = 4 * self.input_dim

        task_design_space = self._set_space(design_space_info)
        self.design_space = GPyOpt.Design_space(space=task_design_space)
        if search_space_info is not None:
            task_search_space = self._set_space(search_space_info)
        else:
            task_search_space = self._set_default_search_space()
        self.search_space = GPyOpt.Design_space(space=task_search_space)

    def _to_searchspace(self, X: Union[ConfigSpace.Configuration, Dict]) -> Dict:
        xx = []
        search_bound_dic = self._get_var_bound('search')
        search_bounds =[]
        design_bound_dic = self._get_var_bound('design')
        design_bounds = []
        search_type_dic = self._get_var_type('search')
        search_type = []

        for  k, v in X.items():
            xx.append(v)
            search_bounds.append(search_bound_dic[k])
            design_bounds.append(design_bound_dic[k])
            search_type.append(search_type_dic[k])

        xx = np.array(xx)
        search_bounds = np.array(search_bounds)
        design_bounds = np.array(design_bounds)
        xx = (xx - design_bounds[:, 0]) * (search_bounds[:, 1] - search_bounds[:, 0]) / (design_bounds[:, 1] - design_bounds[:, 0]) + (search_bounds[:, 0])

        int_flag = [idx for idx, i in enumerate(search_type) if i == 'discrete']

        configuration_t = {k: np.round(xx[idx]).astype(int) if idx in int_flag else xx[idx] for idx, k in
                           enumerate(design_bound_dic.keys())}

        return configuration_t

    def _to_designspace(self, X: Union[ConfigSpace.Configuration, Dict]) -> Dict:
        xx = []
        search_bound_dic = self._get_var_bound('search')
        search_bounds =[]
        design_bound_dic = self._get_var_bound('design')
        design_bounds = []
        design_type_dic = self._get_var_type('design')
        design_type = []
        for  k, v in X.items():
            xx.append(v)
            search_bounds.append(search_bound_dic[k])
            design_bounds.append(design_bound_dic[k])
            design_type.append(design_type_dic[k])

        xx = np.array(xx)
        search_bounds = np.array(search_bounds)
        design_bounds = np.array(design_bounds)
        xx = (xx - search_bounds[:, 0]) * (design_bounds[:, 1] - design_bounds[:, 0]) / (search_bounds[:, 1] - search_bounds[:, 0]) + (design_bounds[:, 0])

        int_flag = [idx for idx, i in enumerate(design_type) if i == 'discrete']

        configuration_t = {k: np.round(xx[idx]).astype(int) if idx in int_flag else xx[idx] for idx, k in
                           enumerate(design_bound_dic.keys())}
        return configuration_t

    def transform(self, X: Union[ConfigSpace.Configuration, Dict, List[Union[ConfigSpace.Configuration, Dict]]]) -> Union[Dict, List[Dict]]:
        """
        Transform configurations from design_space to search_space.

        :param X: Configurations in design_space.
        :return: Transformed configurations in search_space.
        """

        if isinstance(X, list):
            return [self._to_searchspace(x) for x in X]
        else:
            return self._to_searchspace(X)

    def inverse_transform(self, X: Union[ConfigSpace.Configuration, Dict, List[Union[ConfigSpace.Configuration, Dict]]]) -> Union[Dict, List[Dict]]:
        """
        Transform configurations from search_space back to design_space.

        :param X: Configurations in search_space.
        :return: Transformed configurations back in design_space.
        """
        if isinstance(X, list):
            return [self._to_designspace(x) for x in X]
        else:
            return self._to_designspace(X)

    def observe(self, input_vectors: Union[List[Dict], Dict], output_value: Union[List[Dict], Dict]) -> None:
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
        self._Y = np.vstack((self._Y, output_to_ndarray(output_value))) if self._Y.size else output_to_ndarray(output_value)


    @abc.abstractmethod
    def reset(self, design_space: Dict, search_sapce: Union[None, Dict] = None):
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