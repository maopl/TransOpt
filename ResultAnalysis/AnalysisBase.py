import abc
from dataclasses import dataclass
from typing import Dict, Hashable, Tuple, List
import numpy as np
from collections import defaultdict
import json
from Util.Data import vectors_to_ndarray, output_to_ndarray
from Knowledge_Base.KnowledgeBase import KnowledgeBase


@dataclass
class result():
    def __init__(self):
        self.X = None
        self.Y = None
        self.best_X = None
        self.best_Y = None



class AnalysisBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, Exper_folder, Methods, Seeds, Tasks = None, start = None, end = None):
        self._Exper_folder = Exper_folder
        self._Methods = Methods
        self._Seeds = Seeds
        self._Tasks = Tasks
        self._init = start
        self._end = end
        self.results = {}

    def read_data_from_kb(self):
        for method in self._Methods:
            self.results[method] = defaultdict(dict)
            for Seed in self._Seeds:
                self.results[method][Seed] = defaultdict(dict)


                file_path = '{}/{}/{}_KB.json'.format(self._Exper_folder, method, Seed)
                database = KnowledgeBase(file_path)

                for dataset_id in database.get_all_dataset_id():
                    r = result()
                    dataset = database.get_dataset_by_id(dataset_id)
                    input_vector = dataset['input_vector']
                    output_value = dataset['output_value']
                    task_name = dataset['name']
                    r.X = vectors_to_ndarray(dataset['dataset_info']['variable_name'], input_vector)
                    r.Y = output_to_ndarray(output_value)
                    best_id = np.argmin(r.Y)
                    r.best_Y = r.Y[best_id]
                    r.best_X = r.X[best_id]
                    self.results[method][Seed][task_name] = r

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

