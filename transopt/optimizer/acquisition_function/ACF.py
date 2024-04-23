# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import scipy
import numpy as np

from GPyOpt.util.general import get_quantiles
from GPyOpt.util.mcmc_sampler import AffineInvariantEnsembleSampler
from GPyOpt.util import epmgp
from GPyOpt.core.task.cost import constant_cost_withGradients

class AcquisitionBase(object):
    """
    Base class for acquisition functions in Bayesian Optimization

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer

    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, cost_withGradients=None):
        self.model = model
        self.space = space
        self.optimizer = optimizer
        self.analytical_gradient_acq = self.analytical_gradient_prediction and self.model.analytical_gradient_prediction # flag from the model to test if gradients are available

        if cost_withGradients is  None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()

    def acquisition_function(self,x):
        """
        Takes an acquisition and weights it so the domain and cost are taken into account.
        """
        f_acqu = self._compute_acq(x)
        cost_x, _ = self.cost_withGradients(x)
        x_z = x if self.space.model_dimensionality == self.space.objective_dimensionality else self.space.zip_inputs(x)
        return -(f_acqu*self.space.indicator_constraints(x_z))/cost_x


    def acquisition_function_withGradients(self, x):
        """
        Takes an acquisition and it gradient and weights it so the domain and cost are taken into account.
        """
        f_acqu,df_acqu = self._compute_acq_withGradients(x)
        cost_x, cost_grad_x = self.cost_withGradients(x)
        f_acq_cost = f_acqu/cost_x
        df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        x_z = x if self.space.model_dimensionality == self.space.objective_dimensionality else self.space.zip_inputs(x)
        return -f_acq_cost*self.space.indicator_constraints(x_z), -df_acq_cost*self.space.indicator_constraints(x_z)

    def optimize(self, duplicate_manager=None):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """
        if not self.analytical_gradient_acq:
            out = self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager)
        else:
            out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients, duplicate_manager=duplicate_manager)
        return out

    def _compute_acq(self,x):

        raise NotImplementedError('')

    def _compute_acq_withGradients(self, x):

        raise NotImplementedError('')

class MTAcquisitionEI(AcquisitionBase):
    """
    Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, task_id, optimizer=None, cost_withGradients=None, jitter=0.0001):
        self.optimizer = optimizer
        super(MTAcquisitionEI, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        self.task_id = task_id

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return MTAcquisitionEI(model, space, optimizer, cost_withGradients, jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x, self.task_id)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x, self.task_id)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu


class MTAcquisitionLCB(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function with constant exploration weight.
    See:

    Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
    Srinivas et al., Proc. International Conference on Machine Learning (ICML), 2010

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, task_id, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(MTAcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight
        self.task_id = task_id

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound
        """
        m, s = self.model.predict(x,self.task_id)
        f_acqu = -m + self.exploration_weight * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x, self.task_id)
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        return f_acqu, df_acqu




class AcquisitionEntropySearch(AcquisitionBase):
    def __init__(self, model, space, task_id, optimizer=None, cost_withGradients=None,
                 num_samples=100, num_representer_points=50,
                 proposal_function=None, burn_in_steps=50):
        """
        Entropy Search acquisition function

        In a nutshell entropy search approximates the
        distribution of the global minimum and tries to decrease its
        entropy. See this paper for more details:
            Hennig and C. J. Schuler
            Entropy search for information-efficient global optimization
            Journal of Machine Learning Research, 13, 2012

        Current implementation does not provide analytical gradients, thus
        DIRECT optimizer is preferred over gradient descent for this acquisition

        Parameters
        ----------
        :param model: GPyOpt class of model
        :param space: GPyOpt class of Design_space
        :param sampler: mcmc sampler for representer points, an instance of util.McmcSampler
        :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
        :param cost_withGradients: function
        :param num_samples: integer determining how many samples to draw for each candidate input
        :param num_representer_points: integer determining how many representer points to sample
        :param proposal_function: Function that defines an unnormalized log proposal measure from which to sample the representer. The default is expected improvement.
        :param burn_in_steps: integer that defines the number of burn-in steps when sampling the representer points
        """
        # if not isinstance(model, GPModel):
        #     raise RuntimeError("The current entropy search implementation supports only GPModel as model")

        self.optimizer = optimizer
        self.task_id = task_id
        self.analytical_gradient_prediction = False
        AcquisitionBase.__init__(self, model, space, optimizer, cost_withGradients=cost_withGradients)

        self.input_dim = self.space.input_dim()

        self.num_repr_points = num_representer_points
        self.burn_in_steps = burn_in_steps
        self.sampler = AffineInvariantEnsembleSampler(space)

        # (unnormalized) density from which to sample representer points
        self.proposal_function = proposal_function
        if self.proposal_function is None:
            bounds = space.get_bounds()
            mi = np.zeros(len(bounds))
            ma = np.zeros(len(bounds))
            for d in range(len(bounds)):
                mi[d] = bounds[d][0]
                ma[d] = bounds[d][1]

            ei = MTAcquisitionEI(model, space, task_id)

            def prop_func(x):
                if len(x.shape) != 1:
                    raise ValueError("Expected a vector, received a matrix of shape {}".format(x.shape))
                if np.all(np.all(mi <= x)) and np.all(np.all(x <= ma)):
                    return np.log(np.clip(ei._compute_acq(x), 0., np.PINF))
                else:
                    return np.NINF

            self.proposal_function = prop_func

        # This is used later to calculate derivative of the stochastic part for the loss function
        # Derived following Ito's Lemma, see for example https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma
        self.W = scipy.stats.norm.ppf(np.linspace(1. / (num_samples + 1),
                                                  1 - 1. / (num_samples + 1),
                                                  num_samples))[np.newaxis, :]

        # Initialize parameters to lazily compute them once needed
        self.repr_points = None
        self.repr_points_log = None
        self.logP = None

    def _update_parameters(self):
        """
        Update parameters of the acquisition required to evaluate the function. In particular:
            * Sample representer points repr_points
            * Compute their log values repr_points_log
            * Compute belief locations logP
        """
        self.repr_points, self.repr_points_log = self.sampler.get_samples(self.num_repr_points, self.proposal_function,
                                                                          self.burn_in_steps)

        if np.any(np.isnan(self.repr_points_log)) or np.any(np.isposinf(self.repr_points_log)):
            raise RuntimeError(
                "Sampler generated representer points with invalid log values: {}".format(self.repr_points_log))

        # Removing representer points that have 0 probability of being the minimum (corresponding to log probability being minus infinity)
        idx_to_remove = np.where(np.isneginf(self.repr_points_log))[0]
        if len(idx_to_remove) > 0:
            idx = list(set(range(self.num_repr_points)) - set(idx_to_remove))
            self.repr_points = self.repr_points[idx, :]
            self.repr_points_log = self.repr_points_log[idx]

        # We predict with the noise as we need to make sure that var is indeed positive definite.
        mu, _ = self.model.predict(self.repr_points,self.task_id)
        # we need a vector
        mu = np.ndarray.flatten(mu)
        var = self.model.predict_covariance(self.repr_points, self.task_id)

        self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = epmgp.joint_min(mu, var, with_derivatives=True)
        # add a second dimension to the array
        self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))

    def _required_parameters_initialized(self):
        """
        Checks if all required parameters are initialized.
        """
        return not (self.repr_points is None or self.repr_points_log is None or self.logP is None)

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError("Not implemented")

    def _compute_acq(self, x):
        # Naming of local variables here follows that in the paper

        if x.shape[1] != self.input_dim:
            message = "Dimensionality mismatch: x should be of size {}, but is of size {}".format(self.input_dim,
                                                                                                  x.shape[1])
            raise ValueError(message)

        if not self._required_parameters_initialized():
            self._update_parameters()

        if x.shape[0] > 1:
            results = np.zeros([x.shape[0], 1])
            for j in range(x.shape[0]):
                results[j] = self._compute_acq(x[[j], :])
            return results

        # Number of belief locations
        N = self.logP.size

        # Evaluate innovation, these are gradients of mean and variance of the repr points wrt x
        # see Method for more details
        dMdx, dVdx = self._innovations(x)

        # The transpose operator is there to make the array indexing equivalent to matlab's
        dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]

        dMdx_squared = dMdx.dot(dMdx.T)
        trace_term = np.sum(np.sum(
            np.multiply(self.dlogPdMudMu, np.reshape(dMdx_squared, (1, dMdx_squared.shape[0], dMdx_squared.shape[1]))),
            2), 1)[:, np.newaxis]

        # Deterministic part of change:
        deterministic_change = self.dlogPdSigma.dot(dVdx) + 0.5 * trace_term
        # Stochastic part of change:
        stochastic_change = (self.dlogPdMu.dot(dMdx)).dot(self.W)
        # Predicted new logP:
        predicted_logP = np.add(self.logP + deterministic_change, stochastic_change)
        max_predicted_logP = np.amax(predicted_logP, axis=0)

        # normalize predictions
        max_diff = max_predicted_logP + np.log(np.sum(np.exp(predicted_logP - max_predicted_logP), axis=0))
        lselP = max_predicted_logP if np.any(np.isinf(max_diff)) else max_diff
        predicted_logP = np.subtract(predicted_logP, lselP)

        # We maximize the information gain
        dHp = np.sum(np.multiply(np.exp(predicted_logP), np.add(predicted_logP, self.repr_points_log)), axis=0)

        dH = np.mean(dHp)
        return dH  # there is another minus in the public function

    def _compute_acq_withGradients(self, x):
        raise NotImplementedError("Analytic derivatives are not supported.")

    def _innovations(self, x):
        """
        Computes the expected change in mean and variance at the representer
        points (cf. Section 2.4 in the paper).


        :param x: candidate for which to compute the expected change in the GP
        :type x: np.array(1, input_dim)

        :return: innovation of mean (without samples) and variance at the representer points
        :rtype: (np.array(num_repr_points, 1), np.array(num_repr_points, num_repr_points))

        """

        '''
        The notation differs from the paper. The representer points
        play the role of x*, the test input x is X. The Omega term is applied
        in the calling function _compute_acq. Another difference is that we
        ignore the noise as in the original Matlab implementation:
        https://github.com/ProbabilisticNumerics/entropy-search/blob/master/matlab/GP_innovation_local.m
        '''

        # Get the standard deviation at x without noise
        _, stdev_x = self.model.predict(x, self.task_id, with_noise=False)

        # Compute the variance between the test point x and the representer points
        sigma_x_rep = self.model.get_covariance_between_points(self.repr_points, x, self.task_id)
        dm_rep = sigma_x_rep / stdev_x

        # Compute the deterministic innovation for the variance
        dv_rep = -dm_rep.dot(dm_rep.T)
        return dm_rep, dv_rep
