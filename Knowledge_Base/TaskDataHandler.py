
from typing import List, Any, Union, Tuple, Dict, Callable
from Knowledge_Base.KnowledgeBase import KnowledgeBase



class OptTaskDataHandler:

    def __init__(self, db: KnowledgeBase, args):
        self.__db = db
        self.args = args

    def _set_dataset_info(self, space_info):
        """
               Supplements the dataset_info based on provided space_info.

               Args:
               - space_info (dict): Description of the problem's space.

               Returns:
               - dict: A supplemented dataset_info.
               """

        # Extract 'input_dim' directly from space_info
        input_dim = space_info['input_dim']
        budget = space_info['budget']
        seed = space_info['seed']
        task_id = space_info['task_id']
        variable_names = []
        variable_types = {}
        variable_bounds = {}
        for key, var in space_info.items():
            if key == 'input_dim' or key == 'budget' or  key == 'seed' or  key == 'task_id':
                continue
            variable_names.append(key)
            variable_types[key] = var['type']
            variable_bounds[key] = var['bounds']

        self.dataset['dataset_info'] = {
            'input_dim': input_dim,
            'budget': budget,
            'seed': seed,
            'task_id': task_id,
            'variable_name': variable_names,
            'variable_type': variable_types,
            'variable_bounds': variable_bounds

        }

    def reset_task(self, task_name, task_space_info:Dict):
        self.dataset_id, self.dataset = self.__db._generate_dataset()
        self.dataset['name'] = task_name
        self._set_dataset_info(task_space_info)


    def syn_database(self):
        required_keys = ['input_dim', 'budget', 'seed', 'task_id']

        # 检查是否所有必要的键都在space_info字典中
        for key in required_keys:
            if key not in self.dataset['dataset_info'] :
                raise ValueError(f"Missing key '{key}' in space_info")

        for dataset_id in self.__db.get_all_dataset_id():
            dataset_name = self.__db.get_dataset_by_id(dataset_id)['name']
            dataset_info = self.__db.get_dataset_info_by_id(dataset_id)

            if self.dataset['name'] == dataset_name and all(dataset_info[key] == self.dataset['dataset_info'][key] for key in required_keys):
                self.dataset = self.__db.get_dataset_by_id(dataset_id)
                self.dataset_id = dataset_id
                return

        self.__db.add_dataset(self.dataset_id, self.dataset)

    def get_observation_num(self):
        return len(self.dataset.get('input_vector'))

    def get_dataset_id(self):
        return self.dataset_id

    def get_input_vectors(self):
        return self.dataset.get('input_vector')

    def get_output_value(self):
        return self.dataset.get('output_value')

    def _validate_input_vector(self, input_vector: Dict) -> bool:
        """
        Validates a given input vector dictionary against the expected structure.

        Args:
        - input_vector (dict): The input vector dictionary to validate.

        Returns:
        - bool: True if valid, raises an exception otherwise.
        """
        # Extract variable names from the current dataset_info
        expected_variable_names = self.dataset['dataset_info']['variable_name']

        # Check if all keys in the input_vector are present in expected_variable_names
        if not all(key in expected_variable_names for key in input_vector.keys()):
            raise ValueError("Some variable names in the input vector are not valid.")

        # Further validations can be added here (e.g., value types, bounds, etc.)

        return True

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

        if not isinstance(output['function_value'], (int, float)):
            raise ValueError("'function_value' should be an int or float.")

        if 'info' not in output:
            raise ValueError("Output value must contain 'info' key.")

        if not isinstance(output['info'], dict):
            raise ValueError("'info' key should contain a dictionary.")

        if 'fidelity' not in output['info']:
            raise ValueError("'info' dictionary must contain 'fidelity' key.")

    def add_observation(self, input_vectors: Union[List[Dict], Dict], output_value: Union[List[Dict], Dict]) -> None:
        """
        Adds new observations to the dataset.

        Args:
        - input_vectors (Union[List[Dict], Dict]): A list of dictionaries or a single dictionary, each containing input vector data.
        - output_value (Union[List[Dict], Dict]): A list of dictionaries or a single dictionary containing output values and associated info.
        """
        # Normalize input_vectors and output_value to lists for consistency
        if not isinstance(input_vectors, list):
            input_vectors = [input_vectors]

        if not isinstance(output_value, list):
            output_value = [output_value]

        # Check if the lengths of input_vectors and output_value match
        if len(input_vectors) != len(output_value):
            raise ValueError("The number of input vectors must match the number of output values.")

        # Validate each input vector
        for iv in input_vectors:
            self._validate_input_vector(iv)

        for ov in output_value:
            self._validate_output_value(ov)

        # Add to dataset
        self.dataset['input_vector'].extend(input_vectors)
        self.dataset['output_value'].extend(output_value)

        self._flush_dataset()

    def _flush_dataset(self):
        self.__db.update_dataset(self.dataset_id, self.dataset['name'], self.dataset)


    def update_dataset_info(self, **kwargs: Any) -> None:
        """
        Update the 'dataset_info' dictionary for the dataset with the given dataset_id with the provided key-value pairs.

        Args:
            **kwargs: Arbitrary keyword arguments representing the key-value pairs to add to 'dataset_info'.

        Raises:
            ValueError: If no dataset with the given dataset_id is found.
        """

        # Retrieve the 'dataset_info' dictionary
        dataset_info = self.dataset.get('dataset_info', {})

        # Update the dictionary with the provided key-value pairs
        dataset_info.update(kwargs)

        # Save the updated 'dataset_info' back to the database
        self.dataset['dataset_info'] = dataset_info

    def get_auxillary_data(self):
        pass



