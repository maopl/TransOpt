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
from GPy.mappings.constant import Constant
from Model.MPGP import MPGP
from Model.GP import PriorGP
from Optimizer.BaseModule import OptimizerBase
from Model.PracGP import PracGP
from Model.MOKernel import MOKernel
from External.transfergpbo import models
from External.transfergpbo.models import TaskData

from emukit.core import ContinuousParameter
from emukit.core import ParameterSpace

from Util.Normalization import Normalize,Normalize_std
from Util import Prior
from typing import Dict, Hashable
from External.transfergpbo.models import (
    WrapperBase,
    MHGP,
    SHGP,
    BHGP,
)

import sobol_seq

def construct_MOGP(meta_data, Target_data, input_dim, output_dim, mf=None, base_kernel ='Matern', Q = 1, rank=2):
    X_list = [data.X for data in meta_data]
    Y_list = [data.Y for data in meta_data]

    X_list.append(Target_data['X'])
    train_Y = Target_data['Y']
    Y_list.append(train_Y)

    X, Y, output_index = util.multioutput.build_XY(X_list, Y_list)

    # Set inference Method
    inference_method = ExactGaussianInference()
    ## Set likelihood
    likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_obj_%s" % j) for y, j in
                        zip(Y, range(output_dim))]
    likelihood = MixedNoise(likelihoods_list=likelihoods_list)

    if base_kernel == 'Matern':
        k = GPy.kern.Matern52(input_dim=input_dim)
    else:
        k = GPy.kern.Matern52(input_dim=input_dim)

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

    obj_model = MPGP(X, Y, K, likelihood, Y_metadata={'output_index': output_index},
                          inference_method=inference_method, mean_function=mf, name=f'OBJ MPGP')
    obj_model[f'mixed_noise.Gaussian_noise.*variance'].constrain_bounded(1e-3, 1e-2)
    obj_model['ICM0.B.kappa'].constrain_fixed(np.zeros(shape=(output_dim,)))
    return obj_model

def get_model(
    model_name: str, space: ParameterSpace, source_data: Dict[Hashable, TaskData], target_data
) -> WrapperBase:
    """Create the model object."""
    model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP:
        model = model_class(space.dimensionality)
    else:
        kernel = GPy.kern.Matern52(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)
    model.meta_fit(source_data)
    model.fit(target_data, optimize=True)
    return model

class MOGPOptimizer(OptimizerBase):
    analytical_gradient_prediction = False  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, Xdim, bounds, kernel='Matern', likelihood=None, mean_function=None, acf_name='EI',
                 optimizer='bfgs',  verbose=True):
        self.kernel = kernel
        self.likelihood = likelihood
        self.Xdim = Xdim
        self.bounds = bounds
        self.acf_name = acf_name
        self.mean_function = mean_function
        self.name = 'TMTGP'
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


    def create_model(self, model_name, History_DATA, GYM_data, Target_data, mf, prior:list=[]):

        source_num = len(History_DATA['Y'])
        gym_num = len(GYM_data['Y'])
        self.output_dim = source_num + gym_num + 1

        meta_data = []
        for i in range(source_num):
            meta_data.append(TaskData(X=History_DATA['X'][i], Y=History_DATA['Y'][i]))
        for i in range(gym_num):
            meta_data.append(TaskData(X=GYM_data['X'][i], Y=GYM_data['Y'][i]))
        if self.output_dim > 1:
            self.model_name = model_name
            self.obj_model = construct_MOGP(meta_data,Target_data, self.Xdim, self.output_dim, mf, base_kernel = 'Matern', Q = 1, rank=self.output_dim)
            try:
                self.obj_model.optimize_restarts(messages=False, num_restarts=1, verbose=False)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')

            if source_num > 0:
                self.cal_conformal_quantile(source_id=0)

        else:
            self.model_name = 'GP'
            if self.kernel == None or self.kernel == 'Matern':
                kern = GPy.kern.Matern52(self.Xdim, ARD=False)
            else:
                kern = GPy.kern.Matern52(self.Xdim, ARD=False)
            X = Target_data['X']
            Y = Target_data['Y']

            self.obj_model = PriorGP(X, Y, kernel=kern, mean_function = mf)
            # self.obj_model = GPy.models.GPRegression(X, train_Y, kernel=kern, mean_function=mf)
            self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-3, 0.5)
            try:
                self.obj_model.optimize_restarts(messages=False, num_restarts=2, verbose=False)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')
            self.qhats = None

        if len(prior) == 0:
            self.prior_list = []
            self.prior_list.append(Prior.Gamma(0.5, 1, 'lengthscale'))
            self.prior_list.append(Prior.Gamma(0.5, 1, 'variance'))
        else:
            self.prior_list = prior

        for i in range(len(self.prior_list)):
            self.obj_model.set_prior(self.prior_list[i])

    def updateModel(self,  History_DATA, GYM_data, Target_data):
        ## Train target model
        source_num = len(History_DATA['Y'])
        gym_num = len(GYM_data['Y'])
        self.output_dim = source_num + gym_num + 1

        if self.model_name == 'MOGP':
            meta_data = []
            for i in range(source_num):
                meta_data.append(TaskData(X=History_DATA['X'][i], Y=History_DATA['Y'][i]))
            for i in range(gym_num):
                meta_data.append(TaskData(X=GYM_data['X'][i], Y=GYM_data['Y'][i]))

            X_list = [data.X for data in meta_data]
            Y_list = [data.Y for data in meta_data]
            X_list.append(Target_data['X'])
            Y = Target_data['Y']
            train_Y = Y
            Y_list.append(train_Y)

            self.set_XY(X_list, Y_list)
            try:
                self.obj_model.optimize_restarts(messages=False, num_restarts=1,
                                                 verbose=self.verbose)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')
            if source_num > 0:
                self.cal_conformal_quantile(source_id=0)

        else:
            X = Target_data['X']
            train_Y = Target_data['Y']

            self.obj_model.set_XY(X, train_Y)

            self.obj_model.optimize_restarts(messages=False, num_restarts=1,
                                             verbose=False)

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


    def predict(self, X, full_cov= False):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.model_name == 'MOGP':
            noise_dict  = {'output_index': np.array([task_id] * X.shape[0])[:,np.newaxis].astype(int)}
            X = np.hstack((X, noise_dict['output_index']))

            m, v = self.obj_model.predict(X, Y_metadata=noise_dict, full_cov=full_cov, include_likelihood=True)
            v = np.clip(v, 1e-10, np.inf)

        else:
            m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return m, v

    def predict_source(self, X, source_id):
        if X.ndim == 1:
            X = X[None,:]

        if self.model_name == 'MOGP':
            noise_dict  = {'output_index': source_id}
            X = np.hstack((X, noise_dict['output_index']))

            m, v = self.obj_model.predict(X, Y_metadata=noise_dict, full_cov=False, include_likelihood=True)
            v = np.clip(v, 1e-10, np.inf)

        else:
            m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return m, v

    def cal_conformal_quantile(self, source_id, sample_size=200):
        caliberate_X = 2 * sobol_seq.i4_sobol_generate(self.Xdim, sample_size) - 1
        rand_index = np.random.choice(range(0, sample_size), size=sample_size, replace=False)
        caliberate_X = caliberate_X[rand_index]
        out_put_index =  np.array([source_id]*sample_size)[:, np.newaxis].astype(int)

        # ms,vs = self.predict_source(caliberate_X, out_put_index)
        # if source_num < 2:
        #     ms_list = np.array([ms])
        #     vars_list = np.array([vs])
        # else:
        #     ms_list = np.array([ms[i*sample_size:(i+1)*sample_size] for i in range(source_num - 1)])
        #     vars_list = np.array([vs[i*sample_size:(i+1)*sample_size] for i in range(source_num - 1)])
        t_m, t_v = self.predict(caliberate_X)
        s = self.posterior_samples_source(caliberate_X, out_put_index)[:,0,:]
        # diff = [np.abs(t_m - arr) for arr in ms_list]
        alpha = 0.1  # 1-alpha is the desired coverage
        scores = np.max(np.abs(t_m - s) / (t_v+0.5), axis=1)[:,np.newaxis]

        self.qhats = np.quantile(scores, np.ceil((sample_size + 1)*(1-alpha))/(sample_size), interpolation='higher')

    def conformal_prediction(self, X):
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.model_name == 'MOGP':
            noise_dict  = {'output_index': np.array([task_id] * X.shape[0])[:,np.newaxis].astype(int)}
            X = np.hstack((X, noise_dict['output_index']))

            m, v = self.obj_model.predict(X, Y_metadata=noise_dict, full_cov=False, include_likelihood=True)
            v = np.clip(v, 1e-10, np.inf)

        else:
            m, v = self.obj_model.predict(X)

        if self.qhats is not None:
            return m, self.qhats * v
        else:
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

    def posterior_samples_f(self,X, size=10):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :returns: set of simulations
        :rtype: np.ndarray (Nnew x D x samples)
        """
        m, v = self.predict(X, return_full=True)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]

    def posterior_samples_source(self, X, out_put_index, size=10):
        fsim = self.posterior_samples_source_f(X, out_put_index=out_put_index, size=size)

        if fsim.ndim == 3:
            for d in range(fsim.shape[1]):
                fsim[:, d] = self.samples(fsim[:, d])
        else:
            fsim = self.samples(fsim)
        return fsim

    def posterior_samples_source_f(self, X, out_put_index, size=10):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :returns: set of simulations
        :rtype: np.ndarray (Nnew x D x samples)
        """
        m, v = self.predict_source(X, out_put_index)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        return sim_one_dim(m.flatten(), np.diag(v[:,0]))[:, np.newaxis, :]

    def posterior_samples(self, X, size=10):
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
        fsim = self.posterior_samples_f(X, size=size)

        if fsim.ndim == 3:
            for d in range(fsim.shape[1]):
                fsim[:, d] = self.samples(fsim[:, d])
        else:
            fsim = self.samples(fsim)
        return fsim

    def get_model_para(self):
        if self.model_name == 'SHGP' or \
                self.model_name == 'HGP' or \
                self.model_name == 'MHGP' or \
                self.model_name == 'BHGP' or \
                self.model_name == 'RPGE':
            lengthscale = 1
            variance = 1
        elif self.model_name == 'MOGP':
            lengthscale = self.obj_model['.*lengthscale'][0]
            variance = self.obj_model['.*Mat52.*variance'][0]
        else:
            lengthscale = self.obj_model['Mat52.*lengthscale'][0]
            variance = self.obj_model['Mat52.*variance'][0]

        return lengthscale, variance

    def get_model_prior(self, prior_name):
        return self.obj_model.get_prior(prior_name)

    def update_prior(self, para, para_name):
        self.obj_model.update_prior(para, para_name)



