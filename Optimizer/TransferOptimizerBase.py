import numpy as np
import GPy
import GPyOpt
import time
import ConfigSpace
from paramz import ObsAr
from typing import Dict, Union, List
from Optimizer.BayesianOptimizerAbs import BOAbs
from Knowledge_Base.KnowledgeBaseAccessor import KnowledgeBaseAccessor
from Util.Data import InputData, TaskData, vectors_to_ndarray, output_to_ndarray
from Util.Register import optimizer_register


class TransferOptimizerBase(BOAbs):
    # 类级变量，用于存储注册的方法
    registered_methods = {
        "find_auxiliary_data": {},
        "data_augmentation": {},
        "update_model": {},
        "optimize_acquisition_function": {}
    }

    @classmethod
    def register_method(cls, step_name, method_name, method):
        """注册方法到指定的步骤。"""
        if step_name in cls.registered_methods:
            cls.registered_methods[step_name][method_name] = method

    def __init__(self, config):
        # config是一个字典，它决定了每个步骤要使用哪个方法
        self.config = config

    def find_auxiliary_data(self, data):
        method_name = self.config.get("find_auxiliary_data")
        if method_name in self.registered_methods["find_auxiliary_data"]:
            return self.registered_methods["find_auxiliary_data"][method_name](data)

    # ... 类似的方法逻辑对于 data_augmentation, update_model, optimize_acquisition_function

    def optimize(self, data):
        self.find_auxiliary_data(data)
        self.data_augmentation(data)
        self.update_model(data)
        self.optimize_acquisition_function(data)