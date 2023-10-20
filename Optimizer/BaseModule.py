import abc



class OptimizerBase(abc.ABCMeta, metaclass=abc.ABCMeta):
    """
    The abstract Model for Bayesian Optimization
    """

    MCMC_sampler = False
    analytical_gradient_prediction = False

    @abc.abstractmethod
    def updateModel(self, Source_data, Target_data):
        "Augment the dataset of the model"
        return

    @abc.abstractmethod
    def predict(self, X):
        "Get the predicted mean and std at X."
        return

    # We keep this one optional
    def predict_withGradients(self, X):
        "Get the gradients of the predicted mean and variance at X."
        return

    @abc.abstractmethod
    def get_fmin(self):
        "Get the minimum of the current model."
        return
