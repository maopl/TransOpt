from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, Hashable, Tuple
import numpy as np
from emukit.core.interfaces import IModel


@dataclass
class InputData:
    X: np.ndarray


@dataclass
class TaskData:
    X: np.ndarray
    Y: np.ndarray


class Model(ABC):
    """Abstract model class."""

    def __init__(self):
        """Initializes base model."""
        self._X = None
        self._Y = None

    @property
    def X(self) -> np.ndarray:
        """Return input data."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Return target data."""
        return self._Y

    @abstractmethod
    def meta_fit(self, metadata: Dict[Hashable, TaskData], **kwargs):
        """Train model on historical data.

        Parameters:
        -----------
        metadata
            Dictionary containing a numerical representation of the meta-data that can
            be used to meta-train a model for each task.
        """
        pass

    @abstractmethod
    def fit(self, data: TaskData, **kwargs):
        """Adjust model parameter to the observation on the new dataset.

        Parameters:
        -----------
        data: TaskData
            Observation data.
        """
        pass

    @abstractmethod
    def predict(self, data: InputData) -> (np.ndarray, np.ndarray):
        """Predict outcomes for a given array of input values.

        Parameters:
        -----------
        data: InputData
            Input data to predict on.

        Returns
        -------
        mu: shape = (n_points, 1)
            Predicted mean for every input
        cov: shape = (n_points, n_points) or (n_points, 1)
            Predicted (co-)variance for every input
        """
        pass
    