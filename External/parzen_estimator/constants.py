from typing import Union

import ConfigSpace.hyperparameters as CSH

import numpy as np


EPS = 1.0e-300
NULL_VALUE = 1 << 30
NumericType = Union[float, int]
SQR2, SQR2PI = np.sqrt(2), np.sqrt(2 * np.pi)

CategoricalHPType = Union[CSH.CategoricalHyperparameter]
NumericalHPType = Union[
    CSH.UniformIntegerHyperparameter,
    CSH.UniformFloatHyperparameter,
    CSH.OrdinalHyperparameter,
]
HPType = Union[CategoricalHPType, NumericalHPType]
config2type = {
    "UniformFloatHyperparameter": float,
    "UniformIntegerHyperparameter": int,
    "OrdinalHyperparameter": float,
}


def uniform_weight(size: int) -> np.ndarray:
    return np.ones(size) / size
