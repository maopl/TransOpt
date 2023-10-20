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
from Model.PracGP import PracGP

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

def construct_MOkernel(input_dim,output_dim, base_kernel = 'RBF', Q = 1, rank=2):
    if base_kernel == 'RBF':
        k = GPy.kern.RBF(input_dim=input_dim)
    else:
        k = GPy.kern.RBF(input_dim=input_dim)
    kernel_list = [k] * Q
    j = 1
    kk = kernel_list[0]
    K = kk.prod(
        GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], rank=rank, W=None, kappa=None,
                               name='B'), name='%s%s' % ('ICM', 0))
    for kernel in kernel_list[1:]:
        K += kernel.prod(
            GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], rank=rank, W=None,
                                   kappa=None, name='B'), name='%s%s' % ('ICM', j))
        j += 1
    return K

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

class MultiTaskOptimizer(OptimizerBase):
    analytical_gradient_prediction = False  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, Xdim, bounds, kernel='RBF', likelihood=None, model_name='MOGP', acf_name='EI',
                 optimizer='bfgs',  verbose=True):
        self.kernel = kernel
        self.likelihood = likelihood
        self.Xdim = Xdim
        self.bounds = bounds
        self.acf_name = acf_name
        self.qhats = None

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
        if self.model_name == 'HGP' or self.model_name == 'SHGP' or self.model_name == 'BHGP':
            self.obj_model = get_model(self.model_name, self.model_space)
            self.obj_model.meta_fit(meta_data)
            self.obj_model.fit(TaskData(Target_data['X'], Target_data['Y']), optimize=True)

        elif self.model_name == 'MOGP':
            X_list = list(Source_data['X'])
            Y_list = list(Source_data['Y'])
            X_list.append(Target_data['X'])
            Y_list.append(Target_data['Y'])
            X, Y, output_index = util.multioutput.build_XY(X_list, Y_list)

            #Set inference Method
            inference_method = ExactGaussianInference()
            ## Set likelihood
            likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_obj_%s" % j) for y, j in
                                zip(Y, range(self.output_dim))]
            likelihood = MixedNoise(likelihoods_list=likelihoods_list)

            kernel = construct_MOkernel(self.Xdim, output_dim=self.output_dim, base_kernel=self.kernel, rank=self.output_dim)
            self.obj_model = MPGP(X, Y, kernel, likelihood, Y_metadata={'output_index': output_index}, inference_method=inference_method, name=f'OBJ MPGP')
            try:
                self.obj_model.optimize_restarts(messages=True, num_restarts=1,  verbose=self.verbose)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')
        else:
            if self.kernel == None or self.kernel == 'RBF':
                kern = GPy.kern.RBF(self.Xdim, ARD=True)
            else:
                kern = GPy.kern.RBF(self.Xdim, ARD=True)
            X = Target_data['X']
            Y = Target_data['Y']

            self.obj_model = GPy.models.GPRegression(X, Y, kernel=kern)
            self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
            try:
                self.obj_model.optimize_restarts(messages=True, num_restarts=1, verbose=self.verbose)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')


    def updateModel(self, Source_data, Target_data):
        ## Train target model
        source_num = len(Source_data['Y'])
        if self.model_name == 'HGP' or \
            self.model_name == 'SHGP' or \
                self.model_name == 'BHGP':

            self.obj_model.fit(TaskData(Target_data['X'], Target_data['Y']), optimize=True)
        elif self.model_name == 'MOGP':
            X_list = list(Source_data['X'])
            Y_list = list(Source_data['Y'])
            X_list.append(Target_data['X'])
            Y_list.append(Target_data['Y'])
            self.set_XY(X_list, Y_list)
            try:
                self.obj_model.optimize_restarts(messages=True, num_restarts=1,
                                                        verbose=self.verbose)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')
        else:
            X = Target_data['X']
            Y = Target_data['Y']
            self.obj_model.set_XY(X, Y)
            try:
                self.obj_model.optimize_restarts(messages=True, num_restarts=1,
                                             verbose=self.verbose)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')

    def resetModel(self, Source_data, Target_data):
        ## Train target model
        pass

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
        task_id = self.output_dim - 1

        if self.model_name == 'WSGP' or \
            self.model_name == 'HGP':
            m, v = self.obj_model.predict(X)

        elif self.model_name == 'MOGP':
            noise_dict  = {'output_index': np.array([task_id] * X.shape[0])[:,np.newaxis].astype(int)}
            X = np.hstack((X, noise_dict['output_index']))

            m, v = self.obj_model.predict(X, Y_metadata=noise_dict, full_cov=False, include_likelihood=True)
            v = np.clip(v, 1e-10, np.inf)

        else:
            m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return m, v

    def var_predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.model_name == 'WSGP' or \
            self.model_name == 'HGP':
            m, v = self.obj_model.predict(X)

        elif self.model_name == 'MOGP':
            noise_dict  = {'output_index': np.array([task_id] * X.shape[0])[:,np.newaxis].astype(int)}
            X = np.hstack((X, noise_dict['output_index']))

            m, v = self.obj_model.predict(X, Y_metadata=noise_dict, full_cov=False, include_likelihood=True)
            v = np.clip(v, 1e-10, np.inf)

        else:
            m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return  v

    def obj_posterior_samples(self, X, sample_size):
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.model_name == 'WSGP' or \
                self.model_name == 'HGP':
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

    def set_XY(self, X=None, Y=None):
        if isinstance(X, list):
            X, _, self.obj_model.output_index = util.multioutput.build_XY(X, None)
        if isinstance(Y, list):
            _, Y, self.obj_model.output_index = util.multioutput.build_XY(Y, Y)

        self.obj_model.update_model(False)
        if Y is not None:
            self.obj_model.Y = ObsAr(Y)
            self.obj_model.Y_normalized = self.obj_model.Y
        if X is not None:
            self.obj_model.X = ObsAr(X)

        self.obj_model.Y_metadata = {'output_index': self.obj_model.output_index, 'trials': np.ones(self.obj_model.output_index.shape)}
        if isinstance(self.obj_model.inference_method, expectation_propagation.EP):
            self.obj_model.inference_method.reset()
        self.obj_model.update_model(True)

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