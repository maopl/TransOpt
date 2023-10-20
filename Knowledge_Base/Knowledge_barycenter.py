
from GPy.inference.latent_function_inference.posterior import PosteriorExact as Posterior
from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag
import numpy as np
from GPy import likelihoods
from . import LatentFunctionInference

log_2_pi = np.log(2*np.pi)


class GP_barycenter():
    def __init__(self, Y, K, mean, likelihood = None):
        self.mu = Y
        self.K = K
        self.mean = mean
        if likelihood is None:
            likelihood = likelihoods.Gaussian(variance=1.)
        else:
            self.likelihood = likelihood


        YYT_factor = Y - mean

        Ky = K.copy()
        diag.add(Ky, 1e-8)

        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), None)

        self.posterior, self._log_marginal_likelihood, self.grad_dict =  Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew, pred_var=self._predictive_variable, full_cov=full_cov)
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)
        return mu, var

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None,
                likelihood=None, include_likelihood=True):

        # Predict the latent function values
        mean, var = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)

        if include_likelihood:
            # now push through likelihood
            if likelihood is None:
                likelihood = self.likelihood
            mean, var = likelihood.predictive_values(mean, var, full_cov,
                                                     Y_metadata=Y_metadata)

        if self.normalizer is not None:
            mean = self.normalizer.inverse_mean(mean)

            # We need to create 3d array for the full covariance matrix with
            # multiple outputs.
            if full_cov & (mean.shape[1] > 1):
                var = self.normalizer.inverse_covariance(var)
            else:
                var = self.normalizer.inverse_variance(var)

        return mean, var










