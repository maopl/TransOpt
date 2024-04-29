import numpy as np
import torch
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from transopt.optimizer.model.model_base import  Model
from transopt.optimizer.model.utils import is_pd, nearest_pd
from transopt.agent.registry import model_registry
from transopt.optimizer.model.rbfn import rbfn, RegressionDataset


@model_registry.register('RBFN')
class RBFN(Model):
    def __init__(
        self,
        max_epoch: int = 50,
        batch_size: int = 1,
        lr: float = 0.01,
        num_centers: int = 5,
        normalize: bool = True,
        **options: dict
    ):
        super().__init__()
        self._max_epoch = max_epoch
        self._batch_size = batch_size
        self._lr = lr
        self._num_centers = num_centers
        self._rbfn_model = None

        self._normalize = normalize
        self._x_normalizer = StandardScaler() if normalize else None
        self._y_normalizer = StandardScaler() if normalize else None

        self._options = options

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

        if self._rbfn_model is None:
            dataset = RegressionDataset(torch.from_numpy(_X), torch.from_numpy(_y))
            self._rbfn_model = rbfn(
                dataset=dataset,
                max_epoch=self._max_epoch,
                batch_size=self._batch_size,
                lr=self._lr,
                num_centers=self._num_centers,
            )
        else:
            dataset = RegressionDataset(torch.from_numpy(_X), torch.from_numpy(_y))
            self._rbfn_model.update_dataset(dataset)
        
        try:
            self._rbfn_model.train()
        except np.linalg.LinAlgError as e:
            print('Error: np.linalg.LinAlgError')

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X[None, :]
        
        Y = self._rbfn_model.predict(X)
        return Y, None