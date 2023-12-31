
from typing import List, Any, Union, Tuple, Dict, Callable

import archived.DataSelection
from transopt.KnowledgeBase import KnowledgeBase
from transopt.KnowledgeBase.DataHandlerBase import DataHandler

class OptTaskDataHandler(DataHandler):

    def __init__(self, db: KnowledgeBase, args):
        super(OptTaskDataHandler, self).__init__(db, args)

    def _validate_input_vector(self, input_vector: Dict) -> bool:
        """
        Validates a given input vector dictionary against the expected structure.

        Args:
        - input_vector (dict): The input vector dictionary to validate.

        Returns:
        - bool: True if valid, raises an exception otherwise.
        """
        # Extract variable names from the current dataset_info
        expected_variable_names = self.dataset['dataset_info']['variables'].keys()

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
        expected_obj_num = self.dataset['dataset_info']['num_objective']
        real_obj_num = 0
        for i in output:
            if 'function_value' in i:
                real_obj_num += 1
        if real_obj_num != expected_obj_num:
            raise ValueError(f"Expected number_objective is {expected_obj_num}, but real objective number is {real_obj_num}.")


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
        self.db.update_dataset(self.dataset_id, self.dataset['name'], self.dataset)
        self.db._save_database()


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
        if self.selector is None:
            return {}
        return  self.selector(self, self.args)



