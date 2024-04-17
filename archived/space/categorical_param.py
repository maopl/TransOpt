import numpy as np
import pandas as pd

from transopt.utils.Register import para_register

from .param import BaseParameter


@para_register("categorical")
class CategoricalParameter(BaseParameter):
    def __init__(self, param_info):
        super().__init__(param_info)
        self.categories = list(param_info["categories"])
        try:
            self._categories_dict = {k:v for v, k in enumerate(self.categories)}
        except TypeError: # there are unhashable types
            self._categories_dict = None

    def sample(self, num_samples=1):
        assert num_samples > 0, "Number of samples must be positive"
        return np.random.choice(self.categories, size=num_samples, replace=True)

    def transform(self, value: np.ndarray) -> np.ndarray:
        if self._categories_dict is not None:
            # Use a dictionary for transformation if all categories are hashable
            vectorized_transform = np.vectorize(self._categories_dict.get)
            return vectorized_transform(value).astype(float)
        else:
            # Use a list-based approach for unhashable types
            return np.array([self.categories.index(v) for v in value]).astype(float)

    def inverse_transform(self, value: np.ndarray) -> np.ndarray:
        return np.array([self.categories[int(v)] for v in value.round().astype(int)])

    @property
    def is_discrete_after_transform(self):
        return True
    
    @property
    def opt_lb(self):
        return 0

    @property
    def opt_ub(self):
        return len(self.categories) - 1
