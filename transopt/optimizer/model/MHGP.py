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

from GPy.kern import RBF

import GPy

@model_register('MHGP')
class MHGP():
    """Stack of Gaussian processes.

    Transfer Learning model based on [Golovin et al: Google Vizier: A Service for
    Black-Box Optimization](https://dl.acm.org/doi/abs/10.1145/3097983.3098043).
    Given a list of source data sets, the
    transfer to the target data set is done by training a separate GP for each data set
    whose prior mean function is the posterior mean function of the previous GP in the
    stack.
    """

    def __init__(self, n_features: int):
        """Initialize the Method.

        Args:
            n_features: Number of input parameters of the data.
            within_model_normalize: Normalize each GP internally. Helpful for
                numerical stability.
        """
        super().__init__()

        self.n_samples = 0
        self.n_features = n_features

        self.source_gps = []
        self.target_gp = None

        # GP on difference between target data and last source data set

    def _compute_residuals(self, X, Y) -> np.ndarray:
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

        if not self.source_gps:
            return Y

        predicted_y = self.predict_posterior_mean(X, idx=len(self.source_gps) - 1)

        residuals = Y - predicted_y

        return residuals

    def _update_meta_data(self, *gps: GPy.models.GPRegression):
        """Cache the meta data after meta training."""
        for gp in gps:
            self.source_gps.append(gp)

    def _meta_fit_single_gp(
        self,
        Data:Dict,
        optimize: bool,
    ) -> GPy.models.GPRegression:
        """Train a new source GP on `data`.

        Args:
            data: The source dataset.
            optimize: Switch to run hyperparameter optimization.

        Returns:
            The newly trained GP.
        """
        X = Data['X']
        Y = Data['Y']
        residuals = self._compute_residuals(X, Y)

        kernel = RBF(self.n_features, ARD=True)
        new_gp = GPy.models.GPRegression(X, residuals, kernel=kernel)
        new_gp['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)

        try:
            new_gp.optimize_restarts(num_restarts=1, verbose=False, robust=True)
        except np.linalg.linalg.LinAlgError as e:
            # break
            print('Error: np.linalg.linalg.LinAlgError')

        return new_gp

    def meta_fit(
        self,
        source_datasets: List[Dict],
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
            assert len(source_datasets) == len(optimize)
        optimize_flag = copy.copy(optimize)

        if isinstance(optimize_flag, bool):
            optimize_flag = [optimize_flag] * len(source_datasets)

        for i, source_data in enumerate(source_datasets):
            new_gp = self._meta_fit_single_gp(
                source_data,
                optimize=optimize_flag[i],
            )
            self._update_meta_data(new_gp)


    def meta_update(self):
        self._update_meta_data(self.target_gp)

    def meta_add(self,
        source_datasets: List[Dict],
        optimize: Union[bool, Sequence[bool]] = True
                 ):

        self.meta_fit(source_datasets, optimize)


    def fit(
        self,
        Data:Dict,
        optimize: bool = False,
    ):
        X = Data['X']
        Y = Data['Y']
        self._X = copy.deepcopy(X)
        self._Y = copy.deepcopy(Y)
        self.n_samples, n_features = self._X.shape
        if self.n_features != n_features:
            raise ValueError("Number of features in model and input data mismatch.")

        kern = GPy.kern.RBF(self.n_features, ARD=False)

        residuals = self._compute_residuals(self._X, self._Y)

        self.target_gp = GPy.models.GPRegression(self._X, residuals, kernel=kern)
        self.target_gp['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)

        try:
            self.target_gp.optimize_restarts(num_restarts=1, verbose=False, robust=True)
        except np.linalg.linalg.LinAlgError as e:
            # break
            print('Error: np.linalg.linalg.LinAlgError')



    def predict(
        self, X, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:

        # returned mean: sum of means of the predictions of all source and target GPs
        mu = self.predict_posterior_mean(X)

        # returned variance is the variance of target GP
        _, var = self.target_gp.predict(X)

        return mu, var

    def set_XY(self, Data:Dict):
        self._X = copy.deepcopy(Data['X'])
        self._Y = copy.deepcopy(Data['Y'])

    def predict_posterior_mean(self, X, idx: int = None) -> np.ndarray:
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

        all_gps = self.source_gps + [self.target_gp]

        if idx is None:  # if None, the target GP is considered
            idx = len(all_gps) - 1

        mu = np.zeros((X.shape[0], 1))
        # returned mean is a sum of means of the predictions of all GPs below idx
        for model in all_gps[: idx + 1]:
            mu_, _ = model.predict(X)
            mu += mu_

        return mu

    # def predict_posterior_covariance(self, x1: InputData, x2: InputData) -> np.ndarray:
    #     """Posterior covariance between two inputs.
    #
    #     Args:
    #         x1: First input to be queried. `shape = (n_points_1, n_features)`
    #         x2: Second input to be queried. `shape = (n_points_2, n_features)`
    #
    #     Returns:
    #         Posterior covariance at `(x1, x2)`. `shape = (n_points_1, n_points_2)`
    #     """
    #     return self.target_gp.predict_posterior_covariance(x1, x2)
