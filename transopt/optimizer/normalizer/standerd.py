import numpy as np

from sklearn.preprocessing import StandardScaler

from transopt.agent.registry import normalizer_registry
from transopt.optimizer.normalizer.normalizer_base import NormalizerBase
    
    
@normalizer_registry.register("Standard")
class Standard_normalizer(NormalizerBase):
    def __init__(self, config):
        self.y_data = []
        self.y_mean = None
        self.y_std = None
        super(Standard_normalizer, self).__init__(config)

    def update(self, Y):
        self.y_data.append(Y)
        combined_data = np.vstack(self.y_data)
        self.y_mean = np.mean(combined_data, axis=0)
        self.y_std = np.std(combined_data, axis=0)

    def clear(self):
        self.y_data = []
        self.y_mean = None
        self.y_std = None

    def transform(self, Y=None):
        if Y is not None:
            Y = (Y - self.y_mean) / self.y_std
        return Y

    def inverse_transform(self, Y=None):
        if Y is not None:
            Y = Y * self.y_std + self.y_mean
        return Y