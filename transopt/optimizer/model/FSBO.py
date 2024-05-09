import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler

from GPy import util

from transopt.optimizer.model.model_base import  Model
from transopt.agent.registry import model_registry


@model_registry.register('FSBO')
class FSBO(Model):
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.training_model = None
        self.obj_model = None
        self.device = 'cpu'
        self.Seed = 42  # 示例种子，实际使用时可以根据需要修改

    def meta_fit(self, metadata, **kwargs):
        model_name = kwargs.get('model_name', 'FSBO_Model')
        self.create_model(model_name, metadata, kwargs.get('target_data'))

    def fit(self, X, Y, **kwargs):
        """Re-train or update the existing model with new target data."""
        Target_data = {'X': X, 'Y': Y}
        self.updateModel(Target_data)

    def predict(self, X):
        X_torch = totorch(X, self.device)
        mu, cov = self.obj_model.predict(X_torch)
        mu = mu[:, np.newaxis]  # Ensure mu is a column vector
        cov = cov[:, np.newaxis]  # Simplify covariance to a column vector
        return mu, cov

    def create_model(self, model_name, Meta_data, Target_data):
        self.model_name = model_name
        source_num = len(Meta_data['Y'])
        self.output_dim = source_num + 1

        checkpoint_path = './External/FSBO/checkpoints/'
        self.training_model = FSBO(input_size=self.X.shape[1], checkpoint_path=checkpoint_path, batch_size=len(Meta_data['X'][0]))
        train_data = {}
        for i in range(source_num):
            train_data[i] = {'X': Meta_data['X'][i], 'y': Meta_data['Y'][i]}
        self.training_model.set_data(train_data=train_data)
        self.training_model.meta_train(epochs=1000)
        log_dir = os.path.join(checkpoint_path, "log.txt"),
        self.obj_model = DeepKernelGP(epochs=1000, input_size=self.X.shape[1], checkpoint=checkpoint_path + f'Seed_{self.Seed}_{source_num+1}', log_dir=log_dir, seed=self.Seed)
        self.obj_model.X_obs, self.obj_model.y_obs = totorch(Target_data['X'], self.device), totorch(Target_data['Y'], self.device).reshape(-1)
        self.obj_model.train()