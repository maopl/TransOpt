import numpy as np
from scipy.stats import *
from scipy.spatial import distance
import GPy
from GPyOpt.acquisitions.base import AcquisitionBase
from Util.Register import acf_register

@acf_register('ParEGO')
class ParEGO(AcquisitionBase):
    def __init__(self, model, space, optimizer, config):
        return
        self.x_bounds = x_bounds
        self.x_train = x_train
        self.y_train = y_train
        self.rho = rho
        self.xi = xi
        self.task_num = y_train.shape[1]
        self.train_num = y_train.shape[0]
        self.f_theta = np.zeros(self.train_num)

    def calc_parego(self):
        """
        get the result of optimized ParEGO
        """
        self.scalarization()
        res = self.EI()
        return res

    def scalarization(self):
        """
        scalarize observed output data
        """
        theta = np.random.random_sample((self.task_num))
        sum_theta = np.sum(theta)
        theta = theta / sum_theta

        theta_f = theta * self.y_train
        max_k = np.max(theta_f, axis=1)
        rho_sum_theta_f = self.rho * np.sum(theta_f, axis=1)
        self.f_theta = max_k + rho_sum_theta_f

    def obj(self, x):
        if np.any(np.all(self.x_train == x, axis=1)):
            return 1.0e5
        else:
            mean, var = self.model.predict(np.atleast_2d(x))
            std = np.sqrt(var[0, 0])
            # mean_inv = (-1) * mean
            current_max = self.f_theta.max()
            # print(current_max)
            Z = (current_max - mean[0, 0] - self.xi) / std
            # print(norm.cdf(Z))
            # print(norm.pdf(Z))
            ei = (-1) * (Z * std) * norm.cdf(Z) + std * norm.pdf(Z)
            # print(ei)
            return ei

    def EI(self):
        """
        construct a GP model for scalarized output data
        applying EI for this model
        """
        kernel = GPy.kern.RBF(self.x_train.shape[1])
        self.model = GPy.models.GPRegression(self.x_train, self.f_theta[:, None], kernel=kernel, normalizer=None)
        self.model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-2)
        self.model['.*rbf.variance'].constrain_fixed(1.0)
        # lengthscaleのboundを決定
        x_dist = distance.cdist(self.x_train, self.x_train)
        median = np.median(x_dist)
        if median == 0:
            lower = 1.0e-3
            upper = 100
        else:
            lower = 1.0e-3 * median
            upper = 100 * median
        self.model['.*rbf.lengthscale'].constrain_bounded(lower, upper)
        self.model.optimize_restarts()
        array_bounds = np.array(self.x_bounds)
        max_bound = np.argmax(array_bounds[:, 0] - array_bounds[:, 1])
        terminate_vol = (0.1 ** self.x_train.shape[1]) / (array_bounds[max_bound, 1] - array_bounds[max_bound, 0])
        res = minimize(self.obj, bounds=self.x_bounds, algmethod=1, volper=terminate_vol)
        return res
