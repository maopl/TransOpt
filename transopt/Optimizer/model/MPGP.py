import numpy as np
from transopt.Optimizer.model.GP import PriorGP
import multiprocessing as mp

log_2_pi = np.log(2*np.pi)

def opt_wrapper(args):
    m = args[0]
    kwargs = args[1]
    return m.optimize(**kwargs)



class MPGP(PriorGP):
    analytical_gradient_prediction = False
    def __init__(self,  X, Y, kernel, likelihood, mean_function=None, inference_method=None, name='MPGP', Y_metadata=None, normalizer=False):
        super(MPGP, self).__init__(X, Y, kernel, likelihood, mean_function=mean_function, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)



    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this Method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This Method is not designed to be called manually, the framework is set up to automatically call this Method upon changes to parameters, if you call
            this Method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.mean_function, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)

        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

    def objective_function(self):
        """
        The objective function for the given algorithm.

        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the objective function here.

        For probabilistic models this is the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your objective to minimize here!
        """

        return -float(self.log_likelihood()) - self.log_prior()
    


    def optimize_restarts(self, num_restarts=5, robust=False, verbose=True, parallel=False, num_processes=None, **kwargs):
        """
        Perform random restarts of the model, and set the model to the best
        seen solution.

        If the robust flag is set, exceptions raised during optimizations will
        be handled silently.  If _all_ runs fail, the model is reset to the
        existing parameter values.

        \*\*kwargs are passed to the optimizer.

        :param num_restarts: number of restarts to use (default 10)
        :type num_restarts: int
        :param robust: whether to handle exceptions silently or not (default False)
        :type robust: bool
        :param parallel: whether to run each restart as a separate process. It relies on the multiprocessing module.
        :type parallel: bool
        :param num_processes: number of workers in the multiprocessing pool
        :type numprocesses: int
        :param max_f_eval: maximum number of function evaluations
        :type max_f_eval: int
        :param max_iters: maximum number of iterations
        :type max_iters: int
        :param messages: whether to display during optimisation
        :type messages: bool

        .. note::

            If num_processes is None, the number of workes in the
            multiprocessing pool is automatically set to the number of processors
            on the current machine.

        """
        initial_length = len(self.optimization_runs)
        initial_parameters = self.optimizer_array.copy()

        if parallel: #pragma: no cover
            try:
                pool = mp.Pool(processes=num_processes)
                obs = [self.copy() for i in range(num_restarts)]
                [obs[i].randomize() for i in range(num_restarts-1)]
                jobs = pool.map(opt_wrapper, [(o,kwargs) for o in obs])
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print("Ctrl+c received, terminating and joining pool.")
                pool.terminate()
                pool.join()

        for i in range(num_restarts):
            try:
                if not parallel:
                    if i > 0:
                        self.randomize()
                    self.optimize(**kwargs)
                else:#pragma: no cover
                    self.optimization_runs.append(jobs[i])

                if verbose:
                    print(("Optimization restart {0}/{1}, f = {2}".format(i + 1, num_restarts, self.optimization_runs[-1].f_opt)))
            except np.linalg.LinAlgError:
                print('psd error restart')
                continue
            except Exception as e:
                if robust:
                    print(("Warning - optimization restart {0}/{1} failed".format(i + 1, num_restarts)))
                else:
                    raise e


        if len(self.optimization_runs) > initial_length:
            # This works, since failed jobs don't get added to the optimization_runs.
            i = np.argmin([o.f_opt for o in self.optimization_runs[initial_length:]])
            self.optimizer_array = self.optimization_runs[initial_length + i].x_opt
        else:
            self.optimizer_array = initial_parameters
        return self.optimization_runs


    def predictive_gradients(self, Xnew, kern=None):
        """
        Compute the derivatives of the predicted latent function with respect
        to X*

        Given a set of points at which to predict X* (size [N*,Q]), compute the
        derivatives of the mean and variance. Resulting arrays are sized:
            dmu_dX* -- [N*, Q ,D], where D is the number of output in this GP
            (usually one).

        Note that this is not the same as computing the mean and variance of
        the derivative of the function!

         dv_dX*  -- [N*, Q],    (since all outputs have the same variance)
        :param X: The points at which to get the predictive gradients
        :type X: np.ndarray (Xnew x self.input_dim)
        :returns: dmu_dX, dv_dX
        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q) ]

        """
        if kern is None:
            kern = self.kern
        mean_jac = np.empty((Xnew.shape[0], Xnew.shape[1], self.output_dim))

        for i in range(self.output_dim):
            mean_jac[:, :, i] = kern.gradients_X(
                self.posterior.woodbury_vector[:, i:i+1].T, Xnew,
                self._predictive_variable)

        # Gradients wrt the diagonal part k_{xx}
        dv_dX = kern.gradients_X_diag(np.ones(Xnew.shape[0]), Xnew)

        # Grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx}
        if self.posterior.woodbury_inv.ndim == 3:
            var_jac = np.empty(dv_dX.shape +
                               (self.posterior.woodbury_inv.shape[2],))
            var_jac[:] = dv_dX[:, :, None]
            for i in range(self.posterior.woodbury_inv.shape[2]):
                alpha = -2.*np.dot(kern.K(Xnew, self._predictive_variable),
                                   self.posterior.woodbury_inv[:, :, i])
                var_jac[:, :, i] += kern.gradients_X(alpha, Xnew,
                                                     self._predictive_variable)
        else:
            var_jac = dv_dX
            alpha = -2.*np.dot(kern.K(Xnew, self._predictive_variable),
                               self.posterior.woodbury_inv)
            var_jac += kern.gradients_X(alpha, Xnew, self._predictive_variable)

        if self.normalizer is not None:
            mean_jac = self.normalizer.inverse_mean(mean_jac) \
                       - self.normalizer.inverse_mean(0.)
            if self.output_dim > 1:
                var_jac = self.normalizer.inverse_covariance(var_jac)
            else:
                var_jac = self.normalizer.inverse_variance(var_jac)

        return mean_jac, var_jac


    def predict_jacobian(self, Xnew, kern=None, full_cov=False):
        """
        Compute the derivatives of the posterior of the GP.

        Given a set of points at which to predict X* (size [N*,Q]), compute the
        mean and variance of the derivative. Resulting arrays are sized:

         dL_dX* -- [N*, Q ,D], where D is the number of output in this GP (usually one).
          Note that this is the mean and variance of the derivative,
          not the derivative of the mean and variance! (See predictive_gradients for that)

         dv_dX*  -- [N*, Q],    (since all outputs have the same variance)
          If there is missing data, it is not implemented for now, but
          there will be one output variance per output dimension.

        :param X: The points at which to get the predictive gradients.
        :type X: np.ndarray (Xnew x self.input_dim)
        :param kern: The kernel to compute the jacobian for.
        :param boolean full_cov: whether to return the cross-covariance terms between
        the N* Jacobian vectors

        :returns: dmu_dX, dv_dX
        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q,(D)) ]
        """
        if kern is None:
            kern = self.kern

        mean_jac = np.empty((Xnew.shape[0],Xnew.shape[1],self.output_dim))

        for i in range(self.output_dim):
            mean_jac[:,:,i] = kern.gradients_X(self.posterior.woodbury_vector[:,i:i+1].T, Xnew, self._predictive_variable)

        dK_dXnew_full = np.empty((self._predictive_variable.shape[0], Xnew.shape[0], Xnew.shape[1]))
        one = np.ones((1,1))
        for i in range(self._predictive_variable.shape[0]):
            dK_dXnew_full[i] = kern.gradients_X(one, Xnew, self._predictive_variable[[i]])

        if full_cov:
            dK2_dXdX = kern.gradients_XX(one, Xnew)
        else:
            dK2_dXdX = kern.gradients_XX_diag(one, Xnew)
            #dK2_dXdX = np.zeros((Xnew.shape[0], Xnew.shape[1], Xnew.shape[1]))
            #for i in range(Xnew.shape[0]):
            #    dK2_dXdX[i:i+1,:,:] = kern.gradients_XX(one, Xnew[i:i+1,:])

        def compute_cov_inner(wi):
            if full_cov:
                var_jac = dK2_dXdX - np.einsum('qnm,msr->nsqr', dK_dXnew_full.T.dot(wi), dK_dXnew_full) # n,s = Xnew.shape[0], m = pred_var.shape[0]
            else:
                var_jac = dK2_dXdX - np.einsum('qnm,mnr->nqr', dK_dXnew_full.T.dot(wi), dK_dXnew_full)
            return var_jac

        if self.posterior.woodbury_inv.ndim == 3: # Missing data:
            if full_cov:
                var_jac = np.empty((Xnew.shape[0],Xnew.shape[0],Xnew.shape[1],Xnew.shape[1],self.output_dim))
                for d in range(self.posterior.woodbury_inv.shape[2]):
                    var_jac[:, :, :, :, d] = compute_cov_inner(self.posterior.woodbury_inv[:, :, d])
            else:
                var_jac = np.empty((Xnew.shape[0],Xnew.shape[1],Xnew.shape[1],self.output_dim))
                for d in range(self.posterior.woodbury_inv.shape[2]):
                    var_jac[:, :, :, d] = compute_cov_inner(self.posterior.woodbury_inv[:, :, d])
        else:
            var_jac = compute_cov_inner(self.posterior.woodbury_inv)
        return mean_jac, var_jac

    def posterior_covariance_between_points(self, X1, X2, Y_metadata=None,
                                            likelihood=None,
                                            include_likelihood=True):
        """
        Computes the posterior covariance between points. Includes likelihood
        variance as well as normalization so that evaluation at (x,x) is consistent
        with model.predict

        :param X1: some input observations
        :param X2: other input observations
        :param Y_metadata: metadata about the predicting point to pass to the
                           likelihood
        :param include_likelihood: Whether or not to add likelihood noise to
                                   the predicted underlying latent function f.
        :type include_likelihood: bool

        :returns:
            cov: posterior covariance, a Numpy array, Nnew x Nnew if
            self.output_dim == 1, and Nnew x Nnew x self.output_dim otherwise.
        """

        cov = self._raw_posterior_covariance_between_points(X1, X2)

        if include_likelihood:
            # Predict latent mean and push through likelihood
            mean, _ = self._raw_predict(X1, full_cov=True)
            if likelihood is None:
                likelihood = self.likelihood
            _, cov = likelihood.predictive_values(mean, cov, full_cov=False,
                                                  Y_metadata=Y_metadata)

        if self.normalizer is not None:
            if self.output_dim > 1:
                cov = self.normalizer.inverse_covariance(cov)
            else:
                cov = self.normalizer.inverse_variance(cov)

        return cov



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
        if self.normalizer is not None:
            m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        if self.output_dim == 1:
            return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]
        else:
            fsim = np.empty((X.shape[0], self.output_dim, size))
            for d in range(self.output_dim):
                if v.ndim == 3:
                    fsim[:, d, :] = sim_one_dim(m[:, d], v[:, :, d])
                else:
                    fsim[:, d, :] = sim_one_dim(m[:, d], v)
        return fsim

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