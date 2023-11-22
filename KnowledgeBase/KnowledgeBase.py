import abc
import json
import os
from functools import wraps
from typing import List, Any, Union, Tuple, Dict
import warnings

class KnowledgeBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, file_path: str, load_mode:bool = False):
        self.file_path = file_path
        self.load_mode = load_mode
        self._initialize_data_file()
        self.data_base = self._load_database()
        self._dataset_id = set()
        self._dataset_name = set()
        self._update_datasets_info()


    def _initialize_data_file(self) -> None:
        directory = os.path.dirname(self.file_path)

        # 检查目录是否存在
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                raise OSError(f"Unable to create directory for file: {e}")

        if self.load_mode:
            # 检查文件是否存在
            if not os.path.exists(self.file_path):
                with open(self.file_path, 'w') as f:
                    json.dump({}, f)
            else:
                # 为了确保文件是一个有效的JSON文件，你可以尝试读取它
                try:
                    with open(self.file_path, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"The file {self.file_path} exists but is not a valid JSON file.")
        else:
            with open(self.file_path, 'w') as f:
                json.dump({}, f)

    def _load_database(self) -> dict:
        with open(self.file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or not data:
                return {}
            return data

    def _save_database(self) -> None:
        # def print_value_types(obj, path=""):
        #     """
        #     Print the type of each value in the dictionary. If the value is iterable (like dictionaries and lists),
        #     it recursively prints the type of each nested item.
        #
        #     Args:
        #     obj: The object (typically a dictionary) to analyze.
        #     path: The path to the current object within the main object.
        #     """
        #     if isinstance(obj, dict):
        #         for key, value in obj.items():
        #             new_path = f"{path}.{key}" if path else str(key)
        #             print(f"Path: {new_path}, Type: {type(value).__name__}")
        #             print_value_types(value, new_path)
        #     elif isinstance(obj, list):
        #         for index, item in enumerate(obj):
        #             new_path = f"{path}[{index}]"
        #             print(f"Path: {new_path}, Type: {type(item).__name__}")
        #             print_value_types(item, new_path)
        #     else:
        #         # Not a dict or list, just print the type of the item
        #         print(f"Path: {path}, Type: {type(obj).__name__}")


        # print_value_types(self.data_base)
        with open(self.file_path, 'w') as f:
            json.dump(self.data_base, f)

    def _update_datasets_info(self) -> None:
        ids_in_file = set()
        names_in_file = set()

        for dataset_id, dataset in self.data_base.items():
            if not isinstance(dataset, dict):
                raise ValueError(
                    f"Expected a dictionary for dataset with ID '{dataset_id}', but found {type(dataset)}.")

            dataset_name = dataset.get('name')
            if not dataset_name:
                raise ValueError(f"Dataset with ID '{dataset_id}' does not have a 'name' key.")

            if dataset_id in ids_in_file:
                raise ValueError(f"Duplicate dataset ID '{dataset_id}' found in file.")
            if dataset_name in names_in_file:
                raise ValueError(f"Duplicate dataset name '{dataset_name}' found in file.")

            ids_in_file.add(dataset_id)
            names_in_file.add(dataset_name)

        self._dataset_id.update(ids_in_file)
        self._dataset_name.update(names_in_file)

    def _generate_dataset(self) -> Tuple[int, dict]:
        if not self._dataset_id:
            dataset_id = "1"
        else:
            consecutive_ids = set(str(i) for i in range(1, max(map(int, self._dataset_id)) + 2))
            available_ids = consecutive_ids - self._dataset_id

            if available_ids:
                dataset_id = min(available_ids, key=int)
            else:
                dataset_id = str(max(map(int, self._dataset_id)) + 1)

        # Create a new dataset with empty values
        new_dataset = {
            "name": "",
            "input_vector": [],
            "output_value": [],
            "dataset_info": {}
        }

        return dataset_id, new_dataset

    def select_dataset_by_id(self, dataset_id):
        assert dataset_id in self._dataset_id, "Dataset ID not found."
        return self.data_base[dataset_id]

    def select_dataset_by_name(self, dataset_name):
        for _, dataset in self.data_base.items():
            if dataset["name"] == dataset_name:
                return dataset
        return f"Dataset with name '{dataset_name}' not found."

    def _validate_dataset_info(self, dataset_info: dict) -> None:
        """
        Validates the dataset_info dictionary based on the provided values.

        Args:
        - dataset_info (dict): Dictionary containing the dataset information.

        Returns:
        - None
        """

        # Ensure 'input_dim' is present and of type int
        if 'input_dim' not in dataset_info or not isinstance(dataset_info['input_dim'], int):
            warnings.warn("'input_dim' is either missing or not an integer.")

        # Check 'variable_name'
        if 'variable_name' in dataset_info:
            if not isinstance(dataset_info['variable_name'], list) or not all(
                    isinstance(i, str) for i in dataset_info['variable_name']):
                warnings.warn("'variable_name' should be a list of strings.")
            elif len(dataset_info['variable_name']) != dataset_info['input_dim']:
                warnings.warn("'variable_name' length doesn't match 'input_dim'.")

        # Check 'variable_type'
        if 'variable_type' in dataset_info:
            if not isinstance(dataset_info['variable_type'], list) or not all(
                    i in ['continuous', 'discrete', 'categorical'] for i in dataset_info['variable_type']):
                warnings.warn("'variable_type' should be a list containing 'continuous', 'discrete', or 'categorical'.")
            elif len(dataset_info['variable_type']) != dataset_info['input_dim']:
                warnings.warn("'variable_type' length doesn't match 'input_dim'.")

        # Check 'variable_bounds'
        if 'variable_bounds' in dataset_info and not isinstance(dataset_info['variable_bounds'], list):
            warnings.warn("'variable_bounds' should be a list.")

    def _validate_dataset_structure(self, dataset):
        required_keys = {"name", "input_vector", "output_value", "dataset_info"}
        if not isinstance(dataset, dict):
            raise ValueError("Dataset should be a dictionary.")
        if not required_keys.issubset(dataset.keys()):
            raise ValueError(f"Dataset should contain the keys: {', '.join(required_keys)}")
        if dataset["name"] in self._dataset_name:
            raise ValueError(f"Dataset name '{dataset['name']}' already exists.")

        self._validate_dataset_info(dataset_info=dataset['dataset_info'])


    def add_dataset(self, dataset_id: int, data_set: dict) -> None:
        """
        Adds a new dataset to the knowledge base.

        Parameters:
        - data_set (dict): The dataset to be added. Must contain the keys "name", "input_vector", "output_value", and "dataset_info".

        Returns:
        - None

        Raises:
        - ValueError: If the data_set does not adhere to the required structure or if its name already exists in the knowledge base.
        """

        # Validate the dataset structure
        self._validate_dataset_structure(data_set)

        # Safety check: Ensure that the dataset_id is unique (this should always be true given the design of _generate_id)
        if dataset_id in self.data_base:
            raise ValueError(
                f"Generated dataset ID '{dataset_id}' already exists in the knowledge base. This is unexpected and suggests an issue with ID generation.")

        # Ensure that the dataset name is unique
        if data_set["name"] in self._dataset_name:
            raise ValueError(f"Dataset name '{data_set['name']}' already exists in the knowledge base.")

        # Add the dataset to the internal database
        self.data_base[dataset_id] = data_set

        # Update the sets of dataset IDs and names
        self._dataset_id.add(dataset_id)
        self._dataset_name.add(data_set["name"])

    def delete_dataset(self, dataset_id: int) -> None:
        """
        Deletes a dataset from the knowledge base.

        Parameters:
        - dataset_id (int): The unique ID of the dataset to be deleted.

        Raises:
        - ValueError: If the provided dataset_id does not exist in the knowledge base.
        """

        # Validate that the dataset_id exists in the knowledge base
        if dataset_id not in self.data_base:
            raise ValueError("Dataset ID not found in the knowledge base.")

        # Remove the dataset from the internal database
        del self.data_base[dataset_id]

        # Update the sets of dataset IDs and names
        self._dataset_id.remove(dataset_id)
        self._dataset_name.remove(self.data_base[dataset_id]["name"])


    def update_dataset(self, dataset_id: int = None, dataset_name: str = None, new_dataset: dict = None):
        """
        Updates an existing dataset using the provided dataset_id or dataset_name.

        Args:
        - dataset_id (int, optional): The ID of the dataset to update.
        - dataset_name (str, optional): The name of the dataset to update.
          If both dataset_id and dataset_name are provided, dataset_id will take precedence.
        - new_dataset (dict): The new dataset dictionary to replace the existing one.

        Raises:
        - ValueError: If neither dataset_id nor dataset_name are provided, or if the provided dataset doesn't exist.
        """

        # Validate the arguments
        if not new_dataset:
            raise ValueError("A new dataset dictionary must be provided.")
        if not dataset_id and not dataset_name:
            raise ValueError("Either dataset_id or dataset_name must be provided.")

        # Find the dataset to update using dataset_id or dataset_name
        target_id = None
        if dataset_id:
            if dataset_id in self.data_base:
                target_id = dataset_id
        elif dataset_name:
            for id, data in self.data_base.items():
                if data["name"] == dataset_name:
                    target_id = id
                    break

        if not target_id:
            raise ValueError(f"Dataset with ID {dataset_id} or name '{dataset_name}' not found.")

        # Update the dataset in the database
        self.data_base[target_id] = new_dataset
        self._dataset_name.remove(dataset_name)  # Remove old dataset name
        self._dataset_name.add(new_dataset["name"])  # Add new dataset name

    def get_dataset_by_id(self, dataset_id: str) -> Union[dict, None]:
        """
        Retrieve a dataset from the knowledge base using its ID.

        Args:
            dataset_id (int): The ID of the dataset to retrieve.

        Returns:
            dict: The dataset corresponding to the given ID, if found.
            None: If no dataset with the given ID is found.
        """
        return self.data_base.get(dataset_id, None)

    def get_dataset_info_by_id(self, dataset_id: str) -> Union[dict, None]:
        """
        Retrieve the 'dataset_info' dictionary from the dataset with the given dataset_id.

        Args:
            dataset_id (int): The ID of the dataset to retrieve the 'dataset_info' from.

        Returns:
            dict: The 'dataset_info' dictionary corresponding to the given dataset ID, if found.
            None: If no dataset with the given ID is found.
        """
        dataset = self.data_base.get(dataset_id, {})
        return dataset.get('dataset_info', None)

    def get_input_vectors_by_id(self, dataset_id: str) -> List[Any]:
        """Return the input_vectors from the dataset with the given dataset_id."""
        dataset = self.get_dataset_by_id(dataset_id=dataset_id)
        if not dataset:
            raise ValueError(f"No dataset found with ID {dataset_id}.")

        return dataset.get("input_vector", [])

    def get_output_values_by_id(self, dataset_id: str) -> List[Any]:
        """Return the output_values from the dataset with the given dataset_id."""
        dataset = self.get_dataset_by_id(dataset_id=dataset_id)
        if not dataset:
            raise ValueError(f"No dataset found with ID {dataset_id}.")

        return dataset.get("output_value", [])

    def get_dataset_num(self):
        """
        Return the total number of datasets present in the knowledge base.

        Returns:
            int: The number of datasets.
        """
        return len(self.data_base)

    def get_var_name_by_id(self, dataset_id):
        dataset_info = self.get_dataset_info_by_id(dataset_id)
        assert 'variable_name' in dataset_info
        var_name = dataset_info['variable_name']
        return var_name

    def get_all_dataset_id(self):
        """
        Retrieve all the dataset IDs present in the knowledge base.

        Returns:
            set: A set containing all the dataset IDs.
        """
        return self.data_base.keys()







