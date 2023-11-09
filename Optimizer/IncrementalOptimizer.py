import numpy as np
import GPy
import GPyOpt
import time
from scipy.stats import norm
from GPy import kern
from GPy import util
from paramz import ObsAr

from GPy.inference.latent_function_inference import expectation_propagation
from GPy.inference.latent_function_inference import ExactGaussianInference
from GPy.likelihoods.multioutput_likelihood import MixedNoise


from GPy.core.gp import GP
from Model.MPGP import MPGP
from Optimizer.BaseModule import OptimizerBase
from Optimizer.Model.PracGP import PracGP

from External.transfergpbo import models
from External.transfergpbo.models import TaskData

from emukit.core import ContinuousParameter
from emukit.core import ParameterSpace

from typing import Dict, Hashable
from External.transfergpbo.models import (
    WrapperBase,
    MHGP,
    SHGP,
    BHGP,
)

def get_model(
    model_name: str, space: ParameterSpace
) -> WrapperBase:
    """Create the model object."""
    model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP:
        model = model_class(space.dimensionality)
    else:
        kernel = GPy.kern.RBF(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)

    return model

class IncrementalOptimizer(OptimizerBase):
    analytical_gradient_prediction = False  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, Xdim, bounds, kernel='RBF', likelihood=None, acf_name='EI',
                 optimizer='bfgs',  verbose=True):
        self.kernel = kernel
        self.likelihood = likelihood
        self.Xdim = Xdim
        self.bounds = bounds
        self.acf_name = acf_name

        # Set decision space
        Variables = []
        task_design_space = []
        for var in range(Xdim):
            v_n = f'x{var + 1}'
            Variables.append(ContinuousParameter(v_n, self.bounds[0][var], self.bounds[1][var]))
            var_dic = {'name': f'var_{var}', 'type': 'continuous',
                       'domain': tuple([self.bounds[0][var], self.bounds[1][var]])}
            task_design_space.append(var_dic.copy())
        self.model_space = ParameterSpace(Variables)
        self.acf_space = GPyOpt.Design_space(space=task_design_space)

        self.optimizer = optimizer
        self.verbose = verbose


    def create_model(self, model_name, Source_data, Target_data):
        self.model_name = model_name
        source_num = len(Source_data['Y'])
        self.output_dim = source_num + 1

        ##Meta Date
        meta_data = {}
        for i in range(source_num):
            meta_data[i] = TaskData(X=Source_data['X'][i], Y=Source_data['Y'][i])

        ###Construct objective model
        if self.model_name == 'GP':
            if self.kernel == None or self.kernel == 'RBF':
                kern = GPy.kern.RBF(self.Xdim, ARD=True)
            else:
                kern = GPy.kern.RBF(self.Xdim, ARD=True)
            X = Target_data['X']
            Y = Target_data['Y']

            self.obj_model = GPy.models.GPRegression(X, Y, kernel=kern)
            self.obj_model.optimize_restarts(messages=True, num_restarts=1, verbose=self.verbose)
            self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
            return

        if self.model_name == 'SHGP':
            self.obj_model = get_model('SHGP', self.model_space)
        elif self.model_name == 'MHGP':
            self.obj_model = get_model('MHGP', self.model_space)
        elif self.model_name == 'BHGP':
            self.obj_model = get_model('BHGP', self.model_space)


        self.obj_model.meta_fit(meta_data)

        ## Train target model
        self.obj_model.fit(TaskData(Target_data['X'], Target_data['Y']), optimize=True)



    def updateModel(self, Target_data):
        ## Train target model

        if self.model_name == 'GP':
            X = Target_data['X']
            Y = Target_data['Y']
            self.obj_model.set_XY(X, Y)
            self.obj_model.optimize_restarts(messages=True, num_restarts=1,
                                             verbose=self.verbose)

        if self.model_name == 'SHGP' or \
            self.model_name == 'MHGP' or \
            self.model_name == 'BHGP':

            self.obj_model.fit(TaskData(Target_data['X'], Target_data['Y']), optimize=True)

    def MetaFitModel(self, metadata):

        if self.model_name == 'SHGP' or \
            self.model_name == 'MHGP' or \
            self.model_name == 'BHGP':

            self.obj_model.meta_fit(metadata)

    def meta_add(self, meta_data):
        self.obj_model.meta_add(meta_data)

    def optimize(self):
        time_acf_start = time.time()
        suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
        time_acf_end = time.time()

        suggested_sample = self.model_space.zip_inputs(suggested_sample)

        self.acf_time = time_acf_end - time_acf_start

        print("FitModel:\t{:.0f}h{:.0f}m{:.1f}s".format(
            (self.fit_time)/3600,
            (self.fit_time) % 3600 / 60,
            (self.fit_time) % 3600 % 60,))


        print("AcFun:\t\t{:.0f}h{:.0f}m{:.1f}s".format(
            (self.acf_time)/3600,
            (self.acf_time) % 3600 / 60,
            (self.acf_time) % 3600 % 60,))

        return suggested_sample

    def get_train_time(self):
        return self.fit_time

    def get_fit_time(self):
        return self.acf_time


    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        if X.ndim == 1:
            X = X[None,:]

        # if self.model_name == 'SHGP' or \
        #     self.model_name == 'HGP' or \
        #     self.model_name == 'MHGP' or \
        #     self.model_name == 'BHGP':
        #     m, v = self.obj_model.predict(X)
        # else:
        m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return m, v


    def obj_posterior_samples(self, X, sample_size):
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.model_name == 'SHGP' or \
                self.model_name == 'HGP' or \
                self.model_name == 'MHGP' or \
                self.model_name == 'BHGP' or \
                self.model_name == 'RPGE':
            samples_obj = self.posterior_samples(X, model_id=0,size=sample_size)
        elif self.model_name == 'MOGP':
            noise_dict = {'output_index': np.array([task_id] * X.shape[0])[:, np.newaxis].astype(int)}
            X_zip = np.hstack((X, noise_dict['output_index']))

            samples_obj = self.obj_model.posterior_samples(X_zip, size=sample_size, Y_metadata=noise_dict) # grid * 1 * sample_num

        else:
            raise NameError

        return samples_obj

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self.obj_model.X)

        return m.min()

    def samples(self, gp):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        #orig_shape = gp.shape
        gp = gp.flatten()
        Ysim = np.array([np.random.normal(gpj, scale=np.sqrt(1e-2), size=1) for gpj in gp])
        return Ysim.reshape(orig_shape)

    def posterior_samples_f(self,X, model_id, size=10):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :returns: set of simulations
        :rtype: np.ndarray (Nnew x D x samples)
        """
        m, v = self.obj_model.predict(X, return_full=True)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]


    def posterior_samples(self, X, model_id, size=10):
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
        fsim = self.posterior_samples_f(X, model_id=model_id, size=size)

        if fsim.ndim == 3:
            for d in range(fsim.shape[1]):
                fsim[:, d] = self.samples(fsim[:, d])
        else:
            fsim = self.samples(fsim)
        return fsim