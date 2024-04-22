import copy

import numpy as np
import pandas as pd


class SearchSpace:
    def __init__(self, variables):
        self._variables = {var.name: var for var in variables}
        self.variables_order = [var.name for var in variables]

        # 计算并存储原始范围和类型信息
        self.original_ranges = {
            name: var.search_space_range for name, var in self._variables.items()
        }
        self.var_discrete = {
            name: var.is_discrete for name, var in self._variables.items()
        }

        self.ranges = copy.deepcopy(self.original_ranges)

    def map_to_design_space(self, values: np.ndarray) -> dict:
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise ValueError("values must be a 1D NumPy array.")
        
        values_dict = {}
        for i, name in enumerate(self.variables_order):
            variable = self._variables[name]
            value = values[i]
            values_dict[name] = variable.map_from_search_space(value)
        return values_dict
    
    def map_from_design_space(self, values_dict: dict) -> np.ndarray:
        values_array = np.zeros(len(self.variables_order))
        for i, name in enumerate(self.variables_order):
            variable = self._variables[name]
            value = values_dict[name]
            values_array[i] = variable.map_to_search_space(value)
        return values_array

    def update_range(self, name, new_range: tuple):
        if name in self._variables:
            # Check if the new range is valid
            ori_range = self.original_ranges[name]
            if new_range[0] < ori_range[0] or new_range[1] > ori_range[1]:
                raise ValueError(
                    f"New range {new_range} is out of the original range {ori_range}."
                )
            
            self.ranges[name] = new_range
        else:
            raise ValueError(f"Variable '{name}' not found in search space.")
