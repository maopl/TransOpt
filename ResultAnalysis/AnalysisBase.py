import abc
from dataclasses import dataclass
from typing import Dict, Hashable, Tuple, List, Union
import numpy as np
from collections import defaultdict
import json
from Util.Data import vectors_to_ndarray, output_to_ndarray
from KnowledgeBase.KnowledgeBase import KnowledgeBase
import matplotlib.pyplot as plt
import seaborn as sns
@dataclass
class result():
    """
    Class to store the results of the analysis.
    """
    def __init__(self):
        self.X = None
        self.Y = None
        self.best_X = None
        self.best_Y = None



class AnalysisBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, Exper_folder, methods, seeds, tasks, start = 0, end = None):
        self._Exper_folder = Exper_folder
        self._methods = methods
        self._seeds = seeds
        self._tasks = tasks
        self._init = start
        self._end = end
        self.results = {}
        self._task_names = set()
        self._colors = self.assign_colors_to_methods()

    def read_data_from_kb(self):
        for method in self._methods:
            self.results[method] = defaultdict(dict)
            for Seed in self._seeds:
                self.results[method][Seed] = defaultdict(dict)

                file_path = '{}/{}/{}_KB.json'.format(self._Exper_folder, method, Seed)
                database = KnowledgeBase(file_path)

                for dataset_id in database.get_all_dataset_id():
                    r = result()
                    dataset = database.get_dataset_by_id(dataset_id)
                    input_vector = dataset['input_vector']
                    output_value = dataset['output_value']
                    task_name = dataset['name']
                    if task_name.split('_')[0] not in self._tasks:
                        continue

                    r.X = vectors_to_ndarray(dataset['dataset_info']['variable_name'], input_vector)
                    r.Y = output_to_ndarray(output_value)
                    if self._end is not None:
                        r.X = r.X[:self._end]
                        r.Y = r.Y[:self._end]
                    else:
                        assert len(r.Y) == len(r.X)
                        self._end = len(r.Y)
                    best_id = np.argmin(r.Y)
                    r.best_Y = r.Y[best_id]
                    r.best_X = r.X[best_id]
                    self.results[method][Seed][task_name] = r
                    self._task_names.add(task_name)

    def save_results_to_json(self, file_path):
        # 将 NumPy 数组转换为列表的函数
        def convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError

        # 将数据转换为 JSON 格式并保存到文件
        with open(file_path, 'w') as f:
            json.dump(self.results, f, default=convert)

    def load_results_from_json(self, file_path):
        # 从列表转换回 NumPy 数组的函数
        def convert(dct):
            if 'type' in dct and dct['type'] == 'ndarray':
                return np.array(dct['value'])
            return dct

        # 从文件加载数据并转换格式
        with open(file_path, 'r') as f:
            self.results = json.load(f, object_hook=convert)

    def get_results_by_order(self, order=None):
        """
        Get results from the nested dictionary based on the specified order.
        Args:
            order (list, optional): The order in which results should be organized.
                Defaults to ["task", "method", "seed"].
        Returns:
            dict: A dictionary of results organized according to the specified order.
        """

        # Retrieve the corresponding category based on the type of the key
        def get_key(key):
            if key == 'task':
                return self._task_names
            elif key == 'method':
                return self._methods
            elif key == 'seed':
                return self._seeds

        # Retrieve the corresponding data from the existing results.
        def get_from_original_results(key_list):
            first_original_key = key_list[order.index('method')]
            second_original_key = key_list[order.index('seed')]
            third_original_key = key_list[order.index('task')]
            return self.results[first_original_key][second_original_key][third_original_key]

        if order is None:
            order = ["task", "method", "seed"]
        else:
            assert len(order) == 3, "order's length must be 3"
            assert all(
                key in order for key in ["task", "method", "seed"]), "order must include 'task', 'method', and 'seed'"

        # Define dictionaries for each level of order
        levels = {key: get_key(key) for key in order}

        new_results = {}
        for first_key in levels[order[0]]:
            new_results[first_key] = defaultdict(dict)
            for second_key in levels[order[1]]:
                new_results[first_key][second_key] = defaultdict(dict)
                for third_key in levels[order[2]]:
                    new_results[first_key][second_key][third_key] = get_from_original_results(
                        [first_key, second_key, third_key])

        return new_results

    def assign_colors_to_methods(self):
        """
        Assign a unique color from Matplotlib's 'tab10' color cycle to each method.

        Args:
        methods (list): A list of method names.

        Returns:
        dict: A dictionary where keys are method names and values are their assigned colors.
        """
        # Using the 'tab10' color cycle from Matplotlib
        rgb_colors = [
            (141, 211, 199),
            (255, 255, 179),
            (190, 186, 218),
            (251, 128, 114),
            (128, 177, 211),
            (253, 180, 98),
            (179, 222, 105),
            (252, 205, 229),
            (217, 217, 217),
            (188, 128, 189),
            (204, 235, 197)
        ]

        color_strings = []
        for rgb in rgb_colors:
            color_str = f"rgb,255:red,{rgb[0]}; green,{rgb[1]}; blue,{rgb[2]}"
            color_strings.append(color_str)

        # Creating a dictionary to store method names and their assigned colors
        method_colors = {}
        for i, method in enumerate(self._methods):
            color_index = i % len(color_strings)  # Cycle through colors if there are more methods than colors
            color = color_strings[color_index]
            method_colors[method] = color

        return method_colors

    def get_color_for_method(self, method:Union[List,str]):
        """
        Get the color(s) associated with a specific method or a list of methods.

        Args:
        method (str or list): The name of the method or a list of method names.

        Returns:
        str or list: The hex color code(s) associated with the method(s).
        """
        if isinstance(method, str):
            if method not in self._colors:
                raise ValueError(f"Method {method} not found in colors dictionary")
            return self._colors[method]

        elif isinstance(method, list):
            colors = []
            for m in method:
                if m not in self._colors:
                    raise ValueError(f"Method {m} not found in colors dictionary")
                colors.append(self._colors[m])
            return colors

        else:
            raise TypeError("Input must be a string or a list of strings")

    def get_methods(self):
        """
        Get the list of methods used in the analysis.

        Returns:
        list: A list of method names.
        """
        return self._methods

    def get_task_names(self):
        """
        Get the list of task names used in the analysis.

        Returns:
        list: A list of task names.
        """
        return self._task_names

    def get_seeds(self):
        """
        Get the list of seeds used in the analysis.

        Returns:
        list: A list of seeds.
        """
        return self._seeds