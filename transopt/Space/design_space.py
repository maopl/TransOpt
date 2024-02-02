import pandas as pd
import numpy as np

import transopt.Space
from transopt.utils.Register import para_registry

class DesignSpace:
    def __init__(self):
        self.paras = {}
        self.para_names = []
        self.continuous_names = []
        self.discrete_names = []

    def parse(self, rec):
        self.paras = {}
        self.para_names = []
        self.continuous_names = []
        self.discrete_names = []

        for item in rec:
            assert item['type'] in para_registry, f"Unknown parameter type: {item['type']}"
            param = para_registry[item['type']](item)
            self.paras[param.name] = param
            if param.is_discrete_after_transform:
                self.discrete_names.append(param.name)
            else :
                self.continuous_names.append(param.name) 

        self.para_names = self.continuous_names + self.discrete_names
        assert len(self.para_names) == len(set(self.para_names)), "Duplicated parameter names found"
        return self

    def sample(self, num_samples=1):
        samples = np.column_stack([self.paras[name].sample(num_samples) for name in self.para_names])
        return pd.DataFrame(samples, columns=self.para_names)

    def transform(self, data: pd.DataFrame):
        transformed_data = np.zeros_like(data.values, dtype=float)
        for i, name in enumerate(self.para_names):
            transformed_data[:, i] = self.paras[name].transform(data.iloc[:, i].to_numpy())
        return transformed_data

    def inverse_transform(self, data: np.ndarray):
        inverse_transformed_data = {}
        for i, name in enumerate(self.para_names):
            inverse_transformed_data[name] = self.paras[name].inverse_transform(data[:, i])
        return pd.DataFrame(inverse_transformed_data)
    