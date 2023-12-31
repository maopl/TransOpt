import abc
from typing import List, Any, Union, Tuple, Dict

from transopt.KnowledgeBase import KnowledgeBase


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
    AUX_DATA_SELEC = {}
    AUX_DATA_AUG = {}

    def __init__(self, db: KnowledgeBase, args):
        self.db = db
        self.args = args
        if not hasattr(args, "selector") or args.selector is None:
            self.selector = None
        elif DataHandler.AUX_DATA_SELEC[args.selector] is not None:
            self.selector = DataHandler.AUX_DATA_SELEC[args.selector]
        else:
            # 处理任务名称不在注册表中的情况
            print(f"Optimizer '{args.selector}' not found in the registry.")
            raise NameError

    def _set_dataset_info(self, space_info):
        """
        Supplements the dataset_info based on provided space_info.

        Args:
        - space_info (dict): Description of the problem's space.

        Returns:
        - dict: A supplemented dataset_info.
        """

        # Extract 'input_dim' directly from space_info
        input_dim = space_info["input_dim"]
        budget = space_info["budget"]
        seed = space_info["seed"]
        workload = space_info["workload"]
        num_objective = space_info["num_objective"]

        self.dataset["dataset_info"] = {
            "input_dim": input_dim,
            "num_objective": num_objective,
            "budget": budget,
            "seed": seed,
            "workload": workload,
            "variables": {}
        }
        for key, var in space_info['variables'].items():
            self.dataset["dataset_info"]['variables'][key] = \
                {"type": var["type"], "bounds": var["bounds"]}


    def reset_task(self, task_name, task_space_info: Dict):
        self.dataset_id, self.dataset = self.db._generate_dataset()
        self.dataset["name"] = task_name
        self._set_dataset_info(task_space_info)
        self.syn_database()

    def syn_database(self):
        required_keys = ["input_dim", "budget", "seed", "num_objective", "variables"]

        for key in required_keys:
            if key not in self.dataset["dataset_info"]:
                raise ValueError(f"Missing key '{key}' in space_info")

        for dataset_id in self.db.get_all_dataset_id():
            dataset_name = self.db.get_dataset_by_id(dataset_id)["name"]
            dataset_info = self.db.get_dataset_info_by_id(dataset_id)

            if self.dataset["name"] == dataset_name and all(
                dataset_info[key] == self.dataset["dataset_info"][key]
                for key in required_keys
            ):
                self.dataset = self.db.get_dataset_by_id(dataset_id)
                self.dataset_id = dataset_id
                return

        self.db.add_dataset(self.dataset_id, self.dataset)

    def get_observation_num(self):
        return len(self.dataset.get("input_vector"))

    def get_dataset_id(self):
        return self.dataset_id

    def get_input_vectors(self):
        return self.dataset.get("input_vector")

    def get_output_value(self):
        return self.dataset.get("output_value")

    @abc.abstractmethod
    def add_observation(
        self,
        input_vectors: Union[List[Dict], Dict],
        output_value: Union[List[Dict], Dict],
    ) -> None:
        return

    @abc.abstractmethod
    def get_auxillary_data(self) -> Dict:
        return {}
