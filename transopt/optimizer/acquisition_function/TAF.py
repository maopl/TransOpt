import copy

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles
import numpy as np
from external.transfergpbo.models import InputData, TaskData
from agent.registry import acf_registry

@acf_registry.register('TAF_P')
class AcquisitionTAF_POE(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, config):
        self.optimizer = optimizer
        super(AcquisitionTAF_POE, self).__init__(model, space, optimizer)
        self.model = model
        if 'jitter' in config['jitter']:
            self.jitter = config['jitter']
        else:
            self.jitter = 0.01

        if 'threshold' in config['threshold']:
            self.threshold = config['threshold']
        else:
            self.threshold = 0

        self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, x):
        n_sample = len(x)
        data = InputData(X=x)
        source_num = len(self.model.obj_model._source_gps)
        n_models = source_num + 1
        acf_ei = np.empty((n_models, n_sample, 1))

        for task_uid in range(source_num):
            m, s = self.model.obj_model._source_gps[task_uid].predict(data)
            # fmin = self.CBOModel.get_valid_fmin()
            _X = InputData(self.model.obj_model._source_gps[task_uid]._X)
            fmin = self.model.obj_model._source_gps[task_uid].predict(_X)[0].min()
            phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
            acf_ei[task_uid] =  s * (u * Phi + phi)
        m,s = self.model.predict(x)
        for task_uid in range(source_num):
            acf_ei[task_uid] = acf_ei[task_uid] * self.model.obj_model._source_gp_weights[task_uid][:, np.newaxis]
        acf_ei[-1] = self.model.obj_model._target_model_weight[:, np.newaxis]
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        acf_ei[-1] = acf_ei[-1] * (s * (u * Phi + phi))
        f_acqu_ei = np.sum(acf_ei, axis=0)

        return f_acqu_ei

    def _compute_acq_withGradients(self, x):
        # --- DEFINE YOUR AQUISITION (TO BE MAXIMIZED) AND ITS GRADIENT HERE HERE
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the
        # values of the acquisition at x. df_acqu_x contains is each row the values of the gradient of the
        # acquisition at each point of x.
        #
        # NOTE: this function is optional. If note available the gradients will be approxiamted numerically.
        raise NotImplementedError()


@acf_registry.register('TAF_M')
class AcquisitionTAF_M(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, Model, space, optimizer, cost_withGradients=None, jitter=0.01, threshold=0.):
        self.optimizer = optimizer
        super(AcquisitionTAF_M, self).__init__(Model, space, optimizer)
        self.Model = Model
        self.jitter = jitter
        self.threshold = threshold
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('EIC acquisition does now make sense with cost at present. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, x):
        n_sample = len(x)
        data = InputData(X=x)
        source_num = len(self.Model.obj_model._source_gps)
        n_models = source_num + 1
        acf_ei = np.empty((n_models, n_sample, 1))

        for task_uid in range(source_num):
            m, s = self.Model.obj_model._source_gps[task_uid].predict(data)
            # fmin = self.CBOModel.get_valid_fmin()
            _X = InputData(self.Model.obj_model._source_gps[task_uid]._X)
            fmin = self.Model.obj_model._source_gps[task_uid].predict(_X)[0].min()
            phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
            acf_ei[task_uid] =  s * (u * Phi + phi) * self.Model.obj_model._source_gp_weights[task_uid]
        m,s = self.Model.predict(x)
        fmin = self.Model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        acf_ei[-1] =  (s * (u * Phi + phi)) * self.Model.obj_model._target_model_weight
        f_acqu_ei = np.sum(acf_ei, axis=0)

        return f_acqu_ei

    def _compute_acq_withGradients(self, x):
        # --- DEFINE YOUR AQUISITION (TO BE MAXIMIZED) AND ITS GRADIENT HERE HERE
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the
        # values of the acquisition at x. df_acqu_x contains is each row the values of the gradient of the
        # acquisition at each point of x.
        #
        # NOTE: this function is optional. If note available the gradients will be approxiamted numerically.
        raise NotImplementedError()
