import numpy as np

import GPy
from GPy.core.gp import GP


from paramz.transformations import Transformation, __fixed__
from functools import reduce
from Util.Prior import *

from GPy.likelihoods import gaussian
log_2_pi = np.log(2*np.pi)

def opt_wrapper(args):
    m = args[0]
    kwargs = args[1]
    return m.optimize(**kwargs)


class PriorGP(GP):
    def __init__(self,  X, Y, kernel, likelihood=None, mean_function=None, inference_method=None, name='MPGP', Y_metadata=None, noise_var=1., normalizer=False):
        if kernel is None:
            kernel = GPy.kern.Matern52(X.shape[1])

        if likelihood is None:
            likelihood = GPy.likelihoods.Gaussian(variance=noise_var)

        super(PriorGP, self).__init__(X, Y, kernel, likelihood, mean_function=mean_function, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)
        self.prior_list = []
        self.prior_num = 0

    def log_prior(self):
        """evaluate the prior"""
        if self.prior_num == 0:
            return 0.
        param_array = self.param_array
        priored_indexes = []
        #evaluate the prior log densities
        log_p = 0
        for prior in self.prior_list:
            param = self[f'.*rbf.{prior.name}'][0]
            for i in range(len(self.flattened_parameters)):
                if self.flattened_parameters[i].name == prior.name and self.flattened_parameters[i][0] == param:
                    priored_indexes.append(i)
            log_p += prior.lnpdf(param)

        # log_p = reduce(lambda a, b: a + b, (p.lnpdf(x[ind]).sum() for p, ind in self.priors.items()), 0)


        #account for the transformation by evaluating the log Jacobian (where things are transformed)
        log_j = 0.
        for c,j in self.constraints.items():
            if not isinstance(c, Transformation):continue
            for jj in j:
                if jj in priored_indexes:
                    log_j += c.log_jacobian(param_array[jj])
        return log_p + log_j

    def _log_prior_gradients(self):
        """evaluate the gradients of the priors"""
        if self.prior_num == 0:
            return 0.
        param_array = self.param_array
        ret = np.zeros(param_array.size)
        priored_indexes = []
        #compute derivate of prior density
        log_p = 0
        for prior in self.prior_list:
            # param = self[f'.*Mat52.{prior.name}'][0]
            param = self[f'.*rbf.{prior.name}'][0]
            for i in range(len(self.flattened_parameters)):
                if self.flattened_parameters[i].name == prior.name and self.flattened_parameters[i][0] == param:
                    priored_indexes.append(i)
            try:
                np.put(ret, priored_indexes[-1], prior.lnpdf_grad(param))
            except:
                print(1)

        # [np.put(ret, ind, p.lnpdf_grad(param_array[ind])) for p, ind in self.priors.items()]
        #add in jacobian derivatives if transformed
        # priored_indexes = np.hstack([i for p, i in self.priors.items()])
        for c,j in self.constraints.items():
            if not isinstance(c, Transformation):continue
            for jj in j:
                if jj in priored_indexes:
                    ret[jj] += c.log_jacobian_grad(param_array[jj])
        return ret

    def set_prior(self, prior:Prior, warning=True):
        prior_name = prior.name
        if prior_name == 'lengthscale' or prior_name == 'variance':
            self.prior_list.append(prior)
            self.prior_num += 1
        else:
            print('Unknown prior name!')
            raise NameError



    def update_prior(self, para, prior_name):
        for prior in self.prior_list:
            if prior_name == prior.name:
                prior.update(para)

    def get_prior(self, prior_name):
        for prior in self.prior_list:
            if prior_name == prior.name:
                return prior

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        return PriorGP(gp.X, gp.Y, gp.kern, gp.Y_metadata, gp.normalizer, gp.likelihood.variance.values, gp.mean_function)

    def to_dict(self, save_data=True):
        model_dict = super(PriorGP,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPRegression"
        return model_dict

    @staticmethod
    def _from_dict(input_dict, data=None):
        import GPy
        input_dict["class"] = "GPy.core.GP"
        m = GPy.core.GP.from_dict(input_dict, data)
        return PriorGP.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)