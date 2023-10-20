import copy

import numpy as np
import GPy
import GPyOpt
import time
import ConfigSpace
from paramz import ObsAr
from Acquisition.ConstructACF import get_ACF
from Acquisition.sequential import Sequential
from typing import Dict, Union, List
from Optimizer.BayesianOptimizerAbs import BOAbs
from Knowledge_Base.KnowledgeBaseAccessor import KnowledgeBaseAccessor
from Util.Data import InputData, TaskData, vectors_to_ndarray, output_to_ndarray
from Util.Register import optimizer_register
from Knowledge_Base.TaskDataHandler import OptTaskDataHandler
from Util.Kernel import construct_multi_objective_kernel
from Model.MPGP import MPGP
from Model.GP import PriorGP

from GPy.inference.latent_function_inference import expectation_propagation
from GPy.inference.latent_function_inference import ExactGaussianInference
from GPy.likelihoods.multioutput_likelihood import MixedNoise


@optimizer_register('LFL')
class LFLOptimizer(BOAbs):
    def __init__(self, config:Dict, **kwargs):
        super(LFLOptimizer, self).__init__(config=config)

        self.init_method = 'LFL'
        self.knowledge_num = 2
        self.ini_quantile = 0.5
        self.anchor_points = None
        self.anchor_num = None
        self.model = None

        if 'ini_num' in config:
            self.ini_num = config['ini_num']
        else:
            self.ini_num = None

        if 'acf' in config:
            self.acf = config['acf']
        else:
            self.acf = 'EI'


    def reset_optimizer(self, design_space:Dict, search_sapce:Union[None,Dict] = None):
        self.set_space(design_space, search_sapce)
        self.model = None
        self.updateModel()
        self.acqusition = get_ACF(self.acf, model=self.model, config=self.config)
        self.evaluator = Sequential(self.acqusition)




    def get_auxillary_data(self):
        if self.get_data_callback:
            auxiliary_data = self.get_data_callback()
            print(f"Optimizer received data: {auxiliary_data}")
            # 根据收到的辅助数据进行进一步的设置...
        else:
            return {}

    def initial_sample(self):
        if self.anchor_points is None:
            self.anchor_num = int(self.ini_quantile * self.ini_num)
            self.anchor_points  = self.random_sample(self.anchor_num)

        random_samples = self.random_sample(self.ini_num - self.anchor_num)
        samples = self.anchor_points.copy()
        samples.extend(random_samples)

        return samples

    def random_sample(self, num_samples: int) -> List[Dict]:
        """
        Initialize random samples.

        :param num_samples: Number of random samples to generate
        :return: List of dictionaries, each representing a random sample
        """
        if self.input_dim is None:
            raise ValueError("Input dimension is not set. Call set_search_space() to set the input dimension.")

        random_samples = []
        for _ in range(num_samples):
            sample = {}
            for var_info in self.search_space.config_space:
                var_name = var_info['name']
                var_domain = var_info['domain']
                # Generate a random floating-point number within the specified range
                random_value = np.random.uniform(var_domain[0], var_domain[1])
                sample[var_name] = random_value
            random_samples.append(sample)

        random_samples = self.inverse_transform(random_samples)
        return random_samples

    def set_auxillary_data(self, aux_data:Union[Dict, List[Dict], None]):
        if aux_data is None:
            self.aux_data = None


    def combine_data(self):


    def suggest(self, n_suggestions:Union[None, int] = None)->List[Dict]:
        if self._X.size == 0:
            suggests = self.initial_sample()
            return suggests
        elif self._X.shape[0] < self.ini_num:
            pass
        else:
            if self.aux_data is not None:
                pass
            else:
                self.aux_data = {}
            # aux_dataset_num, aux_dataset_ids  = self.accessor.invoke_custom_method("MTRankS", self._X, self._Y,
            #                                                          params={'anchor_num': self.anchor_num})
            # if int((self.budget - self.ini_num)*0.4) >= self._X.shape[0] - self.ini_num:
            #     G = Generator.PseudoPointsGenerator(n_components=int(Xdim))
            #     X_generated = G.generate_from_fft(Target_data['X'], Target_data['Y'][:, 0],
            #                                       target_samples=len(Target_data['X']) * 2)
            #     pseudo_Y = G.Set_Y_by_gaussian(Target_data['X'], train_y[:, 0], test_X=X_generated)
            #     GYM_data = {'X': [X_generated], 'Y': [pseudo_Y]}
            self.combine_data()
            self.update_model(Data)
            self.updatePrior()




    def get_fmin(self):
        pass

    def predict(self, X):
        pass
    def create_model(self, model_name, Source_data, Target_data, mf, prior:list=[]):
        self.model_name = model_name
        source_num = len(Source_data['Y'])
        self.output_dim = source_num + 1

        X_list = []
        Y_list = []

        X_list.append(meta_data.X)
        X_list.append(Target_data['X'])

        train_Y = Target_data['Y']
        Y_list.append(meta_data.Y)
        Y_list.append(train_Y)

        X, Y, output_index = util.multioutput.build_XY(X_list, Y_list)

        if self.output_dim > 1:
            K = construct_multi_objective_kernel(self.input_dim, self.output_dim, base_kernel='RBF', Q=1, rank=2)
            inference_method = ExactGaussianInference()
            likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_obj_%s" % j) for y, j in
                                zip(Y, range(self.output_dim))]
            likelihood = MixedNoise(likelihoods_list=likelihoods_list)

            self.obj_model = MPGP(X, Y, K, likelihood, Y_metadata={'output_index': output_index},
                                  inference_method=inference_method, mean_function=mf, name=f'OBJ MPGP')

            self.obj_model['mixed_noise.Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
            # self.obj_model['constmap.C'].constrain_fixed(0)
            self.obj_model['ICM0.B.kappa'].constrain_fixed(np.zeros(shape=(self.output_dim,)))

            try:
                self.obj_model.optimize_restarts(messages=False, num_restarts=2, verbose=False)
                self.var_model.optimize_restarts(messages=False, num_restarts=2, verbose=False)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')

        else:
            if self.kernel == None or self.kernel == 'RBF':
                kern = GPy.kern.RBF(self.Xdim, ARD=False)
            else:
                kern = GPy.kern.RBF(self.Xdim, ARD=False)
            X = Target_data['X']
            Y = Target_data['Y']

            self.obj_model = PriorGP(X, Y, kernel=kern, mean_function = mf)
            self.var_model = self.obj_model
            # self.obj_model = GPy.models.GPRegression(X, train_Y, kernel=kern, mean_function=mf)
            self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
            try:
                self.obj_model.optimize_restarts(messages=False, num_restarts=1, verbose=False)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')

        if len(prior) == 0:
            self.prior_list = []
            self.prior_list.append(Prior.LogGaussian(1, 2, 'lengthscale'))
            self.prior_list.append(Prior.LogGaussian(0.5, 2, 'variance'))
        else:
            self.prior_list = prior

        for i in range(len(self.prior_list)):
            self.obj_model.set_prior(self.prior_list[i])

    def updateModel(self, Data):
        ## Train target model
        assert 'Target' in Data
        target_data = Data['Target']

        if 'History' in Data:
            history_data = Data['']
            source_num = len(history_data['Y'])
        else:
            sou
            source_num = 0

        if self.model_name == 'MOGP':
            X_list = list(Source_data['X'])
            Y_list = list(Source_data['Y'])
            X_list.append(Target_data['X'])
            Y = Target_data['Y']
            train_Y = Y
            Y_list.append(train_Y)

            self.set_XY(X_list, Y_list)
            self.var_model.set_XY(Source_data['X'][0], Source_data['Y'][0])

            try:
                self.obj_model.optimize_restarts(messages=False, num_restarts=1,
                                                 verbose=self.verbose)

                self.var_model.optimize_restarts(messages=False, num_restarts=2,
                                                 verbose=self.verbose)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')



        else:
            X = Target_data['X']
            train_Y = Target_data['Y']

            self.obj_model.set_XY(X, train_Y)

            self.obj_model.optimize_restarts(messages=False, num_restarts=1,
                                             verbose=False)

    def updatePrior(self, priorInfo:Dict):
        pass

    def get_auxillary_dataset(self):
        pass
