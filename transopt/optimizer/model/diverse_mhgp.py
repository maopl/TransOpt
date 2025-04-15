# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import numpy as np
from typing import Dict, Hashable, Union, Sequence, Tuple, List

import GPy
from GPy.kern import RBF
from GPy.kern import Kern, RBF
from transopt.optimizer.model.gp import GP
from transopt.optimizer.model.model_base import Model
from sklearn.ensemble import RandomForestRegressor

from transopt.optimizer.model.tpe import TPE
from transopt.agent.registry import model_registry

@model_registry.register("DMHGP")
class DMHGP(Model):
    """Stack of Gaussian processes.

    Transfer Learning model based on [Golovin et al: Google Vizier: A Service for
    Black-Box Optimization](https://dl.acm.org/doi/abs/10.1145/3097983.3098043).
    Given a list of source data sets, the
    transfer to the target data set is done by training a separate GP for each data set
    whose prior mean function is the posterior mean function of the previous GP in the
    stack.
    """

    def __init__(self,         
        kernel: Kern = None,
        noise_variance: float = 1.0,
        normalize: bool = True,
        **options: dict):
        """Initialize the Method.

        Args:
            n_features: Number of input parameters of the data.
            within_model_normalize: Normalize each GP internally. Helpful for
                numerical stability.
        """
        super().__init__()

        self._normalize = normalize
        self._metadata = []
        self._metadata_info = []
        self.model_name = 'GP'
        self._kernel = kernel
        self._noise_variance = noise_variance
        self.n_samples = 0
        self.n_features = None


        self.source_models = []

        # GP on difference between target data and last source data set
        self.target_model = None

    def _compute_residuals(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Determine the difference between given y-values and the sum of predicted
        values from the models in 'source_gps'.

        Args:
            data: Observation (input and target) data.
                Input data: ndarray, `shape = (n_points, n_features)`
                Target data: ndarray, `shape = (n_points, 1)`

        Returns:
            Difference between observed values and sum of predicted values
            from `source_gps`. `shape = (n_points, 1)`
        """
        if self.n_features != X.shape[1]:
            raise ValueError("Number of features in model and input data mismatch.")

        if not self.source_models:
            return Y

        predicted_y = self.predict_posterior_mean(
            X, idx=len(self.source_models) - 1
        )

        residuals = Y - predicted_y

        return residuals

    def _update_meta_data(self, model):
        """Cache the meta data after meta training."""
        self.source_models.append(model)
        self._metadata.append({'X': self._X, 'Y': self._Y})


    def meta_update(self):
        self._update_meta_data(self.target_model)

    def _meta_fit_single_gp(
        self,
        X : np.ndarray,
        Y : np.ndarray,
        optimize: bool,
    ) -> GP:
        """Train a new source GP on `data`.

        Args:
            data: The source dataset.
            optimize: Switch to run hyperparameter optimization.

        Returns:
            The newly trained GP.
        """
        self.n_features = X.shape[1]
        
        residuals = self._compute_residuals(X, Y)
        
        kernel = RBF(self.n_features, ARD=True)
        new_gp = GP(
            kernel, noise_variance=self._noise_variance
        )
        new_gp.fit(
            X = X,
            Y = residuals,
            optimize = optimize,
        )
        return new_gp

    def meta_fit(
        self,
        source_X : List[np.ndarray],
        source_Y : List[np.ndarray],
        optimize: Union[bool, Sequence[bool]] = True,
    ):
        """Train the source GPs on the given source data.

        Args:
            source_datasets: Dictionary containing the source datasets. The stack of GPs
                are trained on the residuals between two consecutive data sets in this
                list.
            optimize: Switch to run hyperparameter optimization.
        """
        assert isinstance(optimize, bool) or isinstance(optimize, list)
        if isinstance(optimize, list):
            assert len(source_X) == len(optimize)
        optimize_flag = copy.copy(optimize)

        if isinstance(optimize_flag, bool):
            optimize_flag = [optimize_flag] * len(source_X)

        for i in range(len(source_X)):
            new_gp = self._meta_fit_single_gp(
                source_X[i],
                source_Y[i],
                optimize=optimize_flag[i],
            )
            self._update_meta_data(new_gp)



    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        optimize: bool = False,
    ):
        self._X = copy.deepcopy(X)
        self._y = copy.deepcopy(Y)
        self.n_samples, self.n_features = self._X.shape
        if self.n_features is None:
            self.n_features = self.n_features
        elif self.n_features != self.n_features:
            raise ValueError("Number of features in model and input data mismatch.")
        
        residuals = self._compute_residuals(X, Y)
        noise = np.random.normal(0, 0.1, residuals.shape)  # Add small Gaussian noise
        residuals = residuals + noise

        if self.model_name == 'GP':
            kern = GPy.kern.RBF(self.n_features, ARD=False)
            self.target_model = GPy.models.GPRegression(self._X, residuals, kernel=kern)
            self.target_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
            try:
                self.target_model.optimize_restarts(num_restarts=1, verbose=True, robust=True)
            except np.linalg.linalg.LinAlgError as e:
                print('Error: np.linalg.linalg.LinAlgError')

        elif self.model_name == 'RF':
            self.target_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5, min_samples_leaf=1, min_samples_split=2)
            self.target_model.fit(self._X, residuals)
        elif self.model_name == 'TPE':
            # Initialize TPE model
            self.target_model = TPE()
            # Fit TPE model with observed data
            self.target_model.fit(self._X, residuals)
        else:
            raise ValueError(f'Invalid model name: {self.model_name}')
    

    def predict(
        self, X: np.ndarray, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        # returned mean: sum of means of the predictions of all source and target GPs
        mu = self.predict_posterior_mean(X)

        # returned variance is the variance of target GP
        n_models = len(self.source_models)
        n_sample = X.shape[0]

        if return_full == False:
            vars_ = np.empty((n_models, n_sample, 1))
        else:
            vars_ = np.empty((n_models, n_sample, n_sample))
        
        if self.model_name == 'GP':
            _, vars_ = self.target_model.predict(X)
        elif self.model_name == 'RF':
            tree_predictions = np.array([tree.predict(X) for tree in self.target_model.estimators_])
            vars_ = np.var(tree_predictions, axis=0).reshape(-1, 1)
        elif self.model_name == 'TPE':
            _, vars_ = self.target_model.predict(X)
        else:
            raise ValueError(f'Invalid model name: {self.model_name}')

        return mu, vars_

    def predict_posterior_mean(self, X: np.ndarray, idx: int = None) -> np.ndarray:
        """Predict the mean function for given test point(s).

        For `idx=None` returns the same as `self.predict(data)[0]` but avoids the
        overhead coming from predicting the variance. If `idx` is specified, returns
        the sum of all the means up to the `idx`-th GP. Useful for inspecting the inner
        state of the stack.

        Args:
            data: Input data to predict on.
                Data is provided as ndarray with shape = (n_points, n_features).
            idx: Integer of the GP in the stack. Counting starts from the bottom at
                zero. If `None`, the mean prediction of the entire stack is returned.

        Returns:
            Predicted mean for every input. `shape = (n_points, 1)`
        """

        all_models = self.source_models + [self.target_model]

        if idx is None:  # if None, the target GP is considered
            idx = len(all_models) - 1

        mu = np.zeros((X.shape[0], 1))
        # returned mean is a sum of means of the predictions of all GPs below idx
        for model_id, model in enumerate(all_models[: idx + 1]):    
            if self.model_name == 'GP':
                mean, var = model.predict(X)
                mu += mean
            elif self.model_name == 'RF':
                tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
                mu += np.mean(tree_predictions, axis=0).reshape(-1, 1)
            elif self.model_name == 'TPE':
                mu += model.predict(X)[0]
            else:
                raise ValueError(f'Invalid model name: {self.model_name}')

        return mu
    

    
    def get_fmin(self):

        return np.min(self._y)