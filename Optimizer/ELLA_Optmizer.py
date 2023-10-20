import numpy as np
import GPy
import GPyOpt
import time
from GPy import kern
from GPy.util.linalg import pdinv, dpotrs, tdot,dpotri



from GPy.core.gp import GP
from Model.MPGP import MPGP
from Optimizer.BaseModule import OptimizerBase
from scipy.linalg import sqrtm, inv, norm
from sklearn.linear_model import Lasso
from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag

from emukit.core import ContinuousParameter
from emukit.core import ParameterSpace


def initial_ELLA(d =2,k=2, mu = 1, lam = 1, k_init = False):
    ELLA_dic = {}
    ELLA_dic['d'] = d
    ELLA_dic['k'] = k
    ELLA_dic['L'] = np.random.randn(d, k)
    ELLA_dic['A'] = np.zeros((d * k, d * k))
    ELLA_dic['b'] = np.zeros((d * k, 1))
    ELLA_dic['S'] = np.zeros((k, 0))
    ELLA_dic['T'] = 0
    ELLA_dic['mu'] = mu
    ELLA_dic['lam'] = lam
    ELLA_dic['k_init'] = k_init

    return ELLA_dic



def update_ELLA(old_ELLA_dic, new_ELLA_dic):
    old_ELLA_dic['L'] = new_ELLA_dic['L']
    old_ELLA_dic['A'] = new_ELLA_dic['A']
    old_ELLA_dic['b'] = new_ELLA_dic['b']
    old_ELLA_dic['S'] = new_ELLA_dic['S']
    old_ELLA_dic['T'] = new_ELLA_dic['T']



class ELLA_Optimizer(OptimizerBase):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).
    :param mean_function: GPy Mapping to use as the mean function for the GP model (default, None).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self,Xdim, bounds, kernel='RBF', acf_name='EI',
                 optimizer='bfgs',  verbose=True, ARD=False, mean_function=None):
        self.Xdim = Xdim
        self.bounds = bounds
        self.acf_name = acf_name
        self.kernel = kernel
        self.optimizer = optimizer
        self.verbose = verbose
        self.obj_model = None
        self.ARD = ARD
        self.mean_function = mean_function

        self.ELLA_dic =  initial_ELLA()
        self.ELLA_dic['T'] += 1

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
        """
        Creates the model given some input data X and Y.
        """
        self.model_name = model_name

        # --- define kernel
        if self.kernel == None or self.kernel == 'RBF':
            kern = GPy.kern.RBF(self.Xdim, ARD=False)
        else:
            kern = GPy.kern.RBF(self.Xdim, ARD=False)
        X = Target_data['X']
        Y = Target_data['Y']

        self.obj_model = GPy.models.GPRegression(X, Y, kernel=kern)
        self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
        self.obj_model.optimize_restarts(messages=True, num_restarts=1, verbose=self.verbose)


    def reset_target(self, X, Y):
        # train model for target task if it will we used (when at least 1 target
        # task observation exists)
        self.input_dim = X.shape[1]
        if self.kernel == None or self.kernel == 'RBF':
            kern = GPy.kern.RBF(self.Xdim, ARD=False)
        else:
            kern = GPy.kern.RBF(self.Xdim, ARD=False)

        self.obj_model = GPy.models.GPRegression(X, Y, kernel=kern)
        self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
        self.obj_model.optimize_restarts(messages=True, num_restarts=1, verbose=self.verbose)


        task_id = self.ELLA_dic['T'] - 1
        self.update_ELLA(X, Y, task_id)


    def updateModel(self, Target_data):
        """
        Updates the model with new observations.
        """
        X = Target_data['X']
        Y = Target_data['Y']
        self.obj_model.set_XY(X, Y)
        self.obj_model.optimize_restarts(messages=True, num_restarts=1,
                                         verbose=self.verbose)

        task_id = self.ELLA_dic['T'] - 1
        self.update_ELLA(X, Y, task_id)

    def update_ELLA(self, X, Y, task_id):

        theta_t = []
        for para in self.obj_model.flattened_parameters:
            theta_t.append(para[0])

        if self.ARD:
            D_t = self.get_hessian(self.obj_model, X, Y, mode='ARD_ONLY')
            theta_t = np.array(theta_t)
            # D_t = self.get_hessian(single_task_model, X, Y, mode='mix')
        else:
            D_t = self.get_hessian(self.obj_model, X, Y, mode='NO_ARD')
            theta_t = np.array(theta_t[:2])
        # D_t[D_t < 1e-5] = 0
        D_t_sqrt = sqrtm(D_t)


        sparse_encode = Lasso(alpha = self.ELLA_dic['mu'] / (X.shape[0] * 2.0),
                              fit_intercept = False).fit(D_t_sqrt.dot(self.ELLA_dic['L']),
                                                         D_t_sqrt.dot(theta_t.T))
        if self.ELLA_dic['k_init'] and task_id < self.ELLA_dic['k']:
            sparse_coeffs = np.zeros((self.ELLA_dic['k'],))
            sparse_coeffs[task_id] = 1.0
        else:
            sparse_coeffs = sparse_encode.coef_
        self.ELLA_dic['S'] = np.hstack((self.ELLA_dic['S'], np.matrix(sparse_coeffs).T))
        self.ELLA_dic['A'] += np.kron(self.ELLA_dic['S'][:,task_id].dot(self.ELLA_dic['S'][:,task_id].T), D_t)
        self.ELLA_dic['b'] += np.kron(self.ELLA_dic['S'][:,task_id].T, np.mat(theta_t).dot(D_t)).T
        L_vectorized = inv(self.ELLA_dic['A'] / self.ELLA_dic['T'] + self.ELLA_dic['lam'] * np.eye(self.ELLA_dic['d'] * self.ELLA_dic['k'], self.ELLA_dic['d'] * self.ELLA_dic['k'])).dot(self.ELLA_dic['b']) / self.ELLA_dic['T']
        self.L = L_vectorized.reshape((self.ELLA_dic['k'], self.ELLA_dic['d'])).T
        self.revive_dead_components()

        theta_new = self.ELLA_dic['L'].dot(self.ELLA_dic['S'][:, task_id])

        if self.ARD:
            pass
        else:
            self.obj_model['rbf.variance'] = theta_new[0]
            self.obj_model['rbf.lengthscale'] = theta_new[1]



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

    def revive_dead_components(self):
        """ re-initailizes any components that have decayed to 0 """
        for i,val in enumerate(np.sum(self.ELLA_dic['L'], axis = 0)):
            if abs(val) < 10 ** -8:
                self.L[:, i] = np.random.randn(self.ELLA_dic['d'],)

    def get_hessian(self, model, X, Y, mode = 'NO_ARD'):
        """ ELLA requires that each single task learner provide the Hessian
            of the loss function evaluated around the optimal single task
            parameters.  This funciton implements this for the base learners
            that are currently supported """

        K = model.kern.K(X)
        variance = model.likelihood.gaussian_variance()
        diag.add(K, variance + 1e-8)
        Ki, LK, LKi, K_logdet = pdinv(K)

        def Hessian(dk_dtheta_a, dk_dtheta_b, d2k_dtheta_ab):
            part1 = 0.5*Y.T @ (Ki @ dk_dtheta_a @ Ki @ dk_dtheta_b @Ki - Ki @ d2k_dtheta_ab @ Ki + Ki @ dk_dtheta_b @ Ki @ dk_dtheta_a @Ki) @ Y
            part2 = 0.5* np.trace(Ki @ d2k_dtheta_ab - Ki @ dk_dtheta_b @ Ki @ dk_dtheta_a)
            return part1 + part2

        if mode == 'NO_ARD':
            ## only three parameter, maybe
            theta_t = []
            for para in model.flattened_parameters:
                theta_t.append(para[0])
            theta_t = np.array(theta_t)

            r = model.kern._scaled_dist(X, None)
            sigma = np.sqrt(theta_t[0])
            dk_dsdl =K*(r*r)/(model.kern.lengthscale * sigma)
            dk_dsds = 2.0 * K /  (sigma ** 2)
            dk_dldl = K*(-3 * r**2 + r**4)/(model.kern.lengthscale**2)

            dk_ds = 2.0 * K / sigma
            dk_dl = K * r*r / model.kern.lengthscale

            map_obj = map(Hessian, [dk_ds, dk_ds, dk_dl],[dk_ds,dk_dl,dk_dl],[dk_dsds, dk_dsdl,dk_dldl])
            tmp = []
            for i in map_obj:
                tmp.append(i)
            Hessian_mat = np.zeros(shape=(2,2))

            Hessian_mat[0][0] = tmp[0]
            Hessian_mat[0][1] = tmp[1]
            Hessian_mat[1][1] = tmp[2]

            diag_mat = np.diag(np.diag(Hessian_mat))
            Hessian_mat = Hessian_mat + Hessian_mat.T - diag_mat

            return Hessian_mat

        elif mode == 'ARD_ONLY' and model.kern.ARD:
            theta_t = []
            for para_id, para in enumerate(model.flattened_parameters):
                if para_id == 1:
                    for i in para[:,X.shape[1]]:
                        theta_t.append(i)
            theta_t = np.array(theta_t)




        elif mode == 'ARD' and model.kern.ARD:
            theta_t = []
            for para_id, para in enumerate(model.flattened_parameters):
                if para_id == 1:
                    for i in para[:,X.shape[1]]:
                        theta_t.append(i)
                else:
                    theta_t.append(para[0])
            theta_t = np.array(theta_t)




    def _predict(self, X, full_cov, include_likelihood):
        if X.ndim == 1:
            X = X[None,:]
        m, v = self.obj_model.predict(X, full_cov=full_cov, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        return m, v

    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        if X.ndim == 1:
            X = X[None,:]
        m, v = self.obj_model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(v)

    def predict_covariance(self, X, with_noise=True):
        """
        Predicts the covariance matric for points in X.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        _, v = self._predict(X, True, with_noise)
        return v

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        return self.obj_model.predict(self.obj_model.X)[0].min()

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.obj_model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.obj_model.predictive_gradients(X)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))

        return m, np.sqrt(v), dmdx, dsdx

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.obj_model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.obj_model.parameter_names_flat().tolist()

    def get_covariance_between_points(self, x1, x2):
        """
        Given the current posterior, computes the covariance between two sets of points.
        """
        return self.obj_model.posterior_covariance_between_points(x1, x2)

