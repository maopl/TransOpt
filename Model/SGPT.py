#Practical gaussian process
import copy
from typing import Dict, Hashable
import numpy as np
from GPy.kern import Kern
from GPy.kern import RBF
from External.transfergpbo.models import InputData, TaskData, Model, GPBO
import sobol_seq


def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)



def compute_ranking_loss(
    f_samps: np.ndarray,
    target_y: np.ndarray,
    target_model: bool,
) -> np.ndarray:
    """
    Compute ranking loss for each sample from the posterior over target points.
    """
    y_stack = np.tile(target_y.reshape((-1, 1)), f_samps.shape[0]).transpose()
    rank_loss = np.zeros(f_samps.shape[0])
    if not target_model:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )
    else:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < y_stack) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )

    return rank_loss



class SGPT_POE(Model):
    def __init__(
            self,
            n_features: int,
            kernel: Kern = None,
            n_samples: int = 5,
            beta = None,
            Seed: int = 0,
            sampling_mode: str = 'bootstrap',
            within_model_normalize: bool = False,
            weight_dilution_strategy = 'probabilistic',
            number_of_function_evaluations=44,
    ):
        super().__init__()
        # GP on difference between target data and last source data set
        self.n_features = n_features
        self._within_model_normalize = within_model_normalize
        self.n_samples = n_samples
        self._metadata = {}
        self._source_gps = {}
        self._source_gp_weights = {}
        self.sampling_mode = sampling_mode
        self.Seed = Seed
        self.rng = np.random.RandomState(self.Seed)
        self.weight_dilution_strategy = weight_dilution_strategy
        self.number_of_function_evaluations = number_of_function_evaluations

        if kernel is None:
            self.kernel = RBF(self.n_features, ARD=True)
        else:
            self.kernel = kernel
        self._target_model = GPBO(copy.deepcopy(self.kernel), noise_variance=0.1, normalize=self._within_model_normalize)
        self._target_model_weight = 1
    def meta_fit(self, source_datasets: Dict[Hashable, TaskData], **kwargs):
        # metadata, _ = SourceSelection.the_k_nearest(source_datasets)

        self._metadata = source_datasets
        self._source_gps = {}
        for idx, task_data in source_datasets.items():
            model = GPBO(copy.deepcopy(self.kernel), noise_variance=0.1,normalize=self._within_model_normalize)
            model.fit(task_data, optimize=True)
            self._source_gps[idx] = model
        self._calculate_weights()


    def meta_add(self, metadata: Dict[Hashable, TaskData], **kwargs):
        n_meta = len(self._metadata)
        # train model for each base task
        n_models = len(self._source_gps)
        for task_uid, task_data in metadata.items():
            self._metadata[n_meta + task_uid] = task_data
            model = GPBO(copy.deepcopy(self.kernel), noise_variance=0.1,normalize=self._within_model_normalize)
            model.fit(task_data, optimize=True)
            self._source_gps[n_models + task_uid] = model

        self._calculate_weights()

    def reset_target(self):
        if len(self._metadata) == 0:
            raise ValueError("No metadata is found. Forgot to run meta_fit?")

        self._X = None
        self._y = None

        # train model for target task if it will we used (when at least 1 target
        # task observation exists)
        self._target_model = GPBO(copy.deepcopy(self.kernel), noise_variance=0.1,normalize=self._within_model_normalize)
        self._calculate_weights()

    def fit(self, data: TaskData, optimize: bool = False):
        self._X = data.X
        self._Y = data.Y
        self._Data = data
        self._target_model.fit(data, optimize)
        self._calculate_weights()


    def predict(self, data: np.ndarray, return_full: bool = False):
        data = InputData(X=data)
        X_test = data.X
        n_models = len(self._source_gp_weights) + 1

        n_sample = X_test.shape[0]
        means = np.empty((n_models, n_sample, 1))
        weights = np.empty((n_models, n_sample))
        if return_full == False:
            vars_ = np.empty((n_models, n_sample, 1))
        else:
            vars_ = np.empty((n_models, n_sample, n_sample))
        for task_uid, weight in enumerate(self._source_gp_weights):
            means[task_uid], vars_[task_uid] = self._source_gps[task_uid].predict(data)
            weights[task_uid] = self.beta * (1/vars_[task_uid])[:,0]

        means[-1], vars_[-1] = self._target_model.predict(data)
        weights[-1] = self.beta * (1/vars_[-1])[:,0]

        normalized_weights = weights / np.sum(weights, axis=0)
        self._source_gp_weights = [normalized_weights[i] for i in range(len(self._source_gps))]
        self._target_model_weight = normalized_weights[-1]

        mean = np.sum(normalized_weights[:,:,np.newaxis] * means, axis=0)
        var = np.sum(weights, axis=0)[:,np.newaxis]
        return mean, var


    def _calculate_weights(self, alpha: float = 0.0):

        # compute proportion of samples for which each model is best
        source_num = len(self._source_gps)
        self.beta = 1 / (source_num + 1)



    def loss(self, task_uid: int) -> np.ndarray:
        model = self._source_gps[task_uid]
        X = self._X
        y = self._Y
        samples = model.sample(InputData(X), size=self.n_samples, with_noise=True)
        sample_comps = samples[:, np.newaxis, :] < samples
        target_comps = np.tile(y[:, np.newaxis, :] < y, self.n_samples)
        return np.sum(sample_comps ^ target_comps, axis=(1, 0))

    def posterior_samples_f(self,X, size=10, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :returns: set of simulations
        :rtype: np.ndarray (Nnew x D x samples)
        """


        predict_kwargs["full_cov"] = True  # Always use the full covariance for posterior samples.
        m, v = self._raw_predict(X,  **predict_kwargs)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]


    def posterior_samples(self, X, size=10, Y_metadata=None, likelihood=None, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim.)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations,
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """


        fsim = self.posterior_samples_f(X, size, **predict_kwargs)
        if likelihood is None:
            likelihood = self.likelihood
        if fsim.ndim == 3:
            for d in range(fsim.shape[1]):
                fsim[:, d] = likelihood.samples(fsim[:, d], Y_metadata=Y_metadata)
        else:
            fsim = likelihood.samples(fsim, Y_metadata=Y_metadata)
        return fsim




class SGPT_M(Model):
    def __init__(
            self,
            n_features: int,
            kernel: Kern = None,
            n_samples: int = 10,
            pha = 2.0,
            Seed: int = 0,
            within_model_normalize: bool = False,
            number_of_function_evaluations=44,
    ):
        super().__init__()
        # GP on difference between target data and last source data set
        self.n_features = n_features
        self._within_model_normalize = within_model_normalize
        self.n_samples = n_samples
        self._metadata = {}
        self._source_gps = {}
        self._source_gp_weights = {}
        self.pha =pha
        self.Seed = Seed
        self.rng = np.random.RandomState(self.Seed)
        self.number_of_function_evaluations = number_of_function_evaluations
        self.sampling_x = 2 * sobol_seq.i4_sobol_generate(n_features, self.n_samples) - 1

        if kernel is None:
            self.kernel = RBF(self.n_features, ARD=True)
        else:
            self.kernel = kernel
        self._target_model = GPBO(copy.deepcopy(self.kernel), noise_variance=0.1, normalize=self._within_model_normalize)
        self._target_model_weight = 1
    def meta_fit(self, source_datasets: Dict[Hashable, TaskData], **kwargs):

        self._metadata = source_datasets
        self._source_gps = {}
        for idx, task_data in source_datasets.items():
            model = GPBO(copy.deepcopy(self.kernel), noise_variance=0.1,normalize=self._within_model_normalize)
            model.fit(task_data, optimize=True)
            self._source_gps[idx] = model
        self._calculate_weights()


    def meta_add(self, metadata: Dict[Hashable, TaskData], **kwargs):
        n_meta = len(self._metadata)
        # train model for each base task
        n_models = len(self._source_gps)
        for task_uid, task_data in metadata.items():
            self._metadata[n_meta + task_uid] = task_data
            model = GPBO(copy.deepcopy(self.kernel), noise_variance=0.1,normalize=self._within_model_normalize)
            model.fit(task_data, optimize=True)
            self._source_gps[n_models + task_uid] = model

        self._calculate_weights()

    def reset_target(self):
        if len(self._metadata) == 0:
            raise ValueError("No metadata is found. Forgot to run meta_fit?")

        self._X = None
        self._y = None

        # train model for target task if it will we used (when at least 1 target
        # task observation exists)
        self._target_model = GPBO(copy.deepcopy(self.kernel), noise_variance=0.1,normalize=self._within_model_normalize)
        self._calculate_weights()

    def fit(self, data: TaskData, optimize: bool = False):
        self._X = data.X
        self._Y = data.Y
        self._Data = data
        self._target_model.fit(data, optimize)
        self._calculate_weights()


    def predict(self, data: np.ndarray, return_full: bool = False):
        data = InputData(X=data)
        X_test = data.X
        n_models = len(self._source_gp_weights)
        if self._target_model_weight > 0:
            n_models += 1
        n_sample = X_test.shape[0]
        means = np.empty((n_models, n_sample, 1))
        weights = np.empty((n_models, n_sample))
        if return_full == False:
            vars_ = np.empty((n_models, n_sample, 1))
        else:
            vars_ = np.empty((n_models, n_sample, n_sample))
        for task_uid, weight in enumerate(self._source_gp_weights):
            means[task_uid], vars_[task_uid] = self._source_gps[task_uid].predict(data)
            weights[task_uid] = weight
        if self._target_model_weight > 0:
            means[-1], vars_[-1] = self._target_model.predict(data)
            weights[-1] = self._target_model_weight

        weights = weights[:,:,np.newaxis]
        mean = np.sum(weights * means, axis=0)
        return mean, vars_[-1]

    def Epanechnikov_kernel(self, X1, X2):
        t=0.5
        diff_matrix = X1 - X2
        weight = np.linalg.norm(diff_matrix, ord=2) / self.pha
        weight = weight * 0.75*(1.0 - t*t)

        return weight

    def _calculate_weights(self, alpha: float = 0.0):
        if self._X is None:
            weight = 1 / len(self._source_gps)
            self._source_gp_weights = [weight for task_uid in self._source_gps]
            self._target_model_weight = 0
            return

        predictions = []
        for model_idx in range(len(self._source_gps)):
            model = self._source_gps[model_idx]
            predictions.append(model.predict(self._Data)[0].flatten())  # ndarray(n,)


        predictions.append(self._target_model.predict(self._Data)[0].flatten())
        predictions = np.array(predictions)

        bootstrap_indices = self.rng.choice(predictions.shape[1],
                                            size=(self.n_samples, predictions.shape[1]),
                                            replace=True)

        bootstrap_predictions = []
        bootstrap_targets = self._Y[bootstrap_indices].reshape((self.n_samples, len(self._Y)))
        for m in range(len(self._source_gps) + 1):
            bootstrap_predictions.append(predictions[m, bootstrap_indices])

        ranking_losses = np.zeros((len(self._source_gps) + 1, self.n_samples))
        for i in range(len(self._source_gps)):
            for j in range(1, len(self._Y)):
                ranking_losses[i] += np.sum(
                    (
                        ~(roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                    ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
                       ), axis=1

                )
        for j in range(1, len(self._Y)):
            ranking_losses[-1] += np.sum(
                (
                        ~((roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)
                        ^ (roll_col(bootstrap_targets, j) < bootstrap_targets))
                ), axis=1
            )
        total_compare = len(self._Y) *(len(self._Y - 1))
        ranking_loss = np.array(ranking_losses) / total_compare

        weights = [self.Epanechnikov_kernel(ranking_loss[task_uid], ranking_loss[-1]) for task_uid in self._source_gps]
        weights.append(1.0)
        weights = np.array(weights)/np.sum(weights)
        self._source_gp_weights = [weights[task_uid] for task_uid in self._source_gps]
        self._target_model_weight = weights[-1]

    def loss(self, task_uid: int) -> np.ndarray:
        model = self._source_gps[task_uid]
        X = self._X
        y = self._Y
        samples = model.sample(InputData(X), size=self.n_samples, with_noise=True)
        sample_comps = samples[:, np.newaxis, :] < samples
        target_comps = np.tile(y[:, np.newaxis, :] < y, self.n_samples)
        return np.sum(sample_comps ^ target_comps, axis=(1, 0))

    def posterior_samples_f(self,X, size=10, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :returns: set of simulations
        :rtype: np.ndarray (Nnew x D x samples)
        """


        predict_kwargs["full_cov"] = True  # Always use the full covariance for posterior samples.
        m, v = self._raw_predict(X,  **predict_kwargs)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]


    def posterior_samples(self, X, size=10, Y_metadata=None, likelihood=None, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim.)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations,
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """


        fsim = self.posterior_samples_f(X, size, **predict_kwargs)
        if likelihood is None:
            likelihood = self.likelihood
        if fsim.ndim == 3:
            for d in range(fsim.shape[1]):
                fsim[:, d] = likelihood.samples(fsim[:, d], Y_metadata=Y_metadata)
        else:
            fsim = likelihood.samples(fsim, Y_metadata=Y_metadata)
        return fsim
if __name__ == '__main__':
    masks = np.eye(10, dtype=np.bool)