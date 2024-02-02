import abc
from typing import List, Any, Union, Tuple, Dict

from transopt.KnowledgeBase.database import Database


def selector_register(name):
    def decorator(func_or_class):
        if name in DataHandler.AUX_DATA_SELEC:
            raise ValueError(f"Error: '{name}' is already registered.")
        DataHandler.AUX_DATA_SELEC[name] = func_or_class
        return func_or_class

    return decorator


def augmentation_register(name):
    def decorator(func_or_class):
        if name in DataHandler.AUX_DATA_AUG:
            raise ValueError(f"Error: '{name}' is already registered.")
        DataHandler.AUX_DATA_AUG[name] = func_or_class
        return func_or_class

    return decorator


class DataHandler(abc.ABC, metaclass=abc.ABCMeta):
    # AUX_DATA_SELEC = {}
    # AUX_DATA_AUG = {}

    def __init__(self, db: Database, args):
        self.db = db
        self.cur_task_name = None
        self.cur_space_info = None
        # self.args = args
        # if not hasattr(args, "selector") or args.selector is None:
        #     self.selector = None
        # elif DataHandler.AUX_DATA_SELEC[args.selector] is not None:
        #     self.selector = DataHandler.AUX_DATA_SELEC[args.selector]
        # else:
        #     # 处理任务名称不在注册表中的情况
        #     print(f"Optimizer '{args.selector}' not found in the registry.")
        #     raise NameError


    def task_bind(self, task_name:str, space_info: Dict):
        self.task_name = task_name
        space_info = self._check_space_info(space_info)
        self.space_info = space_info
        self._check_task_in_db(task_name, self.space_info)


    def _check_space_info(self, space_info: Dict) -> Dict:
        """
        Check and complete the space_info dictionary by filling in missing keys.

        Args:
            space_info: The space information to be validated and completed.

        Returns:
            Dict: The validated and completed space information.
        """
        required_keys = ['variables',  'objectives', 'fidelity']
        missing_keys = required_keys - space_info.keys()
        if missing_keys:
            raise ValueError(f"Required key(s) {missing_keys} are missing in space_info.")

        self._check_space_info_value(space_info, 'var_names', list(space_info['variables'].keys()))
        self._check_space_info_value(space_info, 'var_num', len(space_info['variables']))
        self._check_space_info_value(space_info, 'obj_names', list(space_info['objectives'].keys()))
        self._check_space_info_value(space_info, 'obj_num', len(space_info['objectives']))
        self._check_space_info_value(space_info, 'fidelity_names', list(space_info['fidelity'].keys()))
        self._check_space_info_value(space_info, 'fidelity_num', len(space_info['fidelity']))

        return space_info

    def _check_space_info_value(self, space_info: Dict, key: str, expected_value):
        if key not in space_info:
            space_info[key] = expected_value
            raise Warning(f"No key '{key}' in space_info, auto filled with {expected_value}.")
        elif space_info[key] != expected_value:
            raise Warning(f"Inconsistency found for key '{key}' in space_info.")

    def create_dataset(self, dataset_name:str, space_info: Dict) -> None:
        """
        Args:
            space_info: {
            "var_names": ["x1", "x2"],
            "var_num": 2,
            "variables": {
                "x1": {"type": "continuous", "range": [-5.12, 5.12], "default": 0.0},
                "x2": {"type": "continuous", "range": [-5.12, 5.12], "default": 0.0},
            },

            "obj_names": ["y1", "y2"],
            "obj_num": 2,
            "objectives": {"y1": {"type": "minimize"}, "y2": {"type": "maximize"}},

            "fidelity_names": ["f1", "f2"],
            "fidelity_num": 2,
            "fidelity": {
                "f1": {"type": "continuous", "range": [0, 1], "default": 0.0},
                "f2": {"type": "continuous", "range": [0, 1], "default": 0.0},
            },
        }

        Returns: None
        """
        self.db.create_table(dataset_name, space_info, overwrite=True)

    def _check_task_in_db(self, task_name:str, space_info: Dict):
        if not self.db.check_table_exist(task_name):
            self.create_dataset(task_name, space_info)
        else:
            Warning(f'Dataset "{task_name}" already exists.')
            stored_config = self.db.query_config(task_name)
            if space_info != stored_config:
                raise ValueError(f"Space info not match for task '{task_name}'.")


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

    def write_data(self, data):
        pass

    def read_data_by_var(self):
        pass

    def read_data_by_obj(self):
        pass

    def read_data_by_fid(self):
        pass

    def read_data_all(self, task_name:str):
        pass


    # def _set_dataset_info(self, space_info):
    #     """
    #     Supplements the dataset_info based on provided space_info.
    #
    #     Args:
    #     - space_info (dict): Description of the problem's space.
    #
    #     Returns:
    #     - dict: A supplemented dataset_info.
    #     """
    #
    #     # Extract 'input_dim' directly from space_info
    #     input_dim = space_info["input_dim"]
    #     budget = space_info["budget"]
    #     seed = space_info["seed"]
    #     workload = space_info["workload"]
    #     num_objective = space_info["num_objective"]
    #
    #     self.dataset["dataset_info"] = {
    #         "input_dim": input_dim,
    #         "num_objective": num_objective,
    #         "budget": budget,
    #         "seed": seed,
    #         "workload": workload,
    #         "variables": {}
    #     }
    #     for key, var in space_info['variables'].items():
    #         self.dataset["dataset_info"]['variables'][key] = \
    #             {"type": var["type"], "bounds": var["bounds"]}


    # def reset_task(self, task_name, task_space_info: Dict):
    #     self.dataset_id, self.dataset = self.db._generate_dataset()
    #     self.dataset["name"] = task_name
    #     self._set_dataset_info(task_space_info)
    #     self.syn_database()

    # def syn_database(self):
    #     required_keys = ["input_dim", "budget", "seed", "num_objective", "variables"]
    #
    #     for key in required_keys:
    #         if key not in self.dataset["dataset_info"]:
    #             raise ValueError(f"Missing key '{key}' in space_info")
    #
    #     for dataset_id in self.db.get_all_dataset_id():
    #         dataset_name = self.db.get_dataset_by_id(dataset_id)["name"]
    #         dataset_info = self.db.get_dataset_info_by_id(dataset_id)
    #
    #         if self.dataset["name"] == dataset_name and all(
    #             dataset_info[key] == self.dataset["dataset_info"][key]
    #             for key in required_keys
    #         ):
    #             self.dataset = self.db.get_dataset_by_id(dataset_id)
    #             self.dataset_id = dataset_id
    #             return
    #
    #     self.db.add_dataset(self.dataset_id, self.dataset)

    def get_data_num(self):
        return self.db.get_num_row(self.task_name)


    @abc.abstractmethod
    def add_observation(
        self,
        input_vectors: Union[List[Dict], Dict],
        output_value: Union[List[Dict], Dict],
    ) -> None:
        return


