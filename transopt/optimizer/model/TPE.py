import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from transopt.optimizer.model.model_base import  Model
from transopt.optimizer.model.utils import is_pd, nearest_pd
from transopt.agent.registry import model_registry


@model_registry.register('TPE')
class TPE(Model):
    def __init__(
        self,
        normalize: bool = True,
        **options: dict
    ):
        super().__init__()

        self._normalize = normalize
        self._x_normalizer = StandardScaler() if normalize else None
        self._y_normalizer = StandardScaler() if normalize else None

        self.model = TPEOptimizer()

    def meta_fit(
        self,
        source_X : List[np.ndarray],
        source_Y : List[np.ndarray],
        **kwargs,
    ):
        pass

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        optimize: bool = True,
    ):
        self._X = np.copy(X)
        self._y = np.copy(Y)
        self._Y = np.copy(Y)

        _X = np.copy(self._X)
        _y = np.copy(self._y)

        if self._normalize:
            _X = self._x_normalizer.fit_transform(_X)
            _y = self._y_normalizer.fit_transform(_y)
    

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass