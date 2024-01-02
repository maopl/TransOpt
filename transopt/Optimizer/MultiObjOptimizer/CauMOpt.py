import numpy as np
import GPy
from typing import Dict, Union, List

from transopt.Optimizer.OptimizerBase import BOBase
from transopt.utils.Register import optimizer_register
from transopt.utils.Normalization import get_normalizer
from transopt.utils.serialization import ndarray_to_vectors,vectors_to_ndarray

def calculate_gini_index(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - sum(probabilities ** 2)
    return gini


def best_feature_by_gini(data, labels):
    # 初始化最低基尼指数和对应的特征
    min_gini = 1
    best_feature = None

    # 遍历每个特征
    for feature_idx in range(data.shape[1]):
        # 获取当前特征下的所有值
        current_feature_values = data[:, feature_idx]

        # 假设的分割方式：基于每个值的基尼指数
        for split_value in np.unique(current_feature_values):
            left_split = labels[current_feature_values <= split_value]
            right_split = labels[current_feature_values > split_value]

            # 计算左右分割的加权基尼指数
            left_gini = calculate_gini_index(left_split)
            right_gini = calculate_gini_index(right_split)
            weighted_gini = (len(left_split) * left_gini + len(right_split) * right_gini) / len(labels)

            # 更新最低基尼指数和对应的特征
            if weighted_gini < min_gini:
                min_gini = weighted_gini
                best_feature = feature_idx

    return best_feature

@optimizer_register("CauMO")
class CauMO(BOBase):
    def __init__(self, config: Dict, **kwargs):
        super(CauMO, self).__init__(config=config)

        self.init_method = "Random"

        if "verbose" in config:
            self.verbose = config["verbose"]
        else:
            self.verbose = True

        if "pop_size" in config:
            self.pop_size = config["pop_size"]
        else:
            self.pop_size = 30
            self.ini_num = self.pop_size

        self.second_space = None
        self.third_space = None


        self.model = []
        self.acf = "CauMOACF"

    def initial_sample(self):
        return self.random_sample(self.ini_num)

    def random_sample(self, num_samples: int) -> List[Dict]:
        """
        Initialize random samples.

        :param num_samples: Number of random samples to generate
        :return: List of dictionaries, each representing a random sample
        """
        if self.input_dim is None:
            raise ValueError(
                "Input dimension is not set. Call set_search_space() to set the input dimension."
            )

        random_samples = []
        for _ in range(num_samples):
            sample = {}
            for var_info in self.search_space.config_space:
                var_name = var_info["name"]
                var_domain = var_info["domain"]
                # Generate a random floating-point number within the specified range
                random_value = np.random.uniform(var_domain[0], var_domain[1])
                sample[var_name] = random_value
            random_samples.append(sample)

        random_samples = self.inverse_transform(random_samples)
        return random_samples

    def update_model(self, Data):
        Target_Data = Data["Target"]
        assert "X" in Target_Data

        X = Target_Data["X"]
        Y = Target_Data["Y"]
        assert Y.shape[0] == self.num_objective

        if self.normalizer is not None:
            Y_norm = np.array([self.normalizer(y) for y in Y])

        if len(self.model_list) == 0:
            self.create_model(X, Y_norm)
        else:
            self.set_data(X, Y_norm)

        try:
            for i in range(len(self.model_list)):
                self.model_list[i].optimize_restarts(
                    num_restarts=1, verbose=self.verbose, robust=True
                )
        except np.linalg.linalg.LinAlgError as e:
            # break
            print("Error: np.linalg.linalg.LinAlgError")

    def create_model(self, X, Y):
        assert self.num_objective is not None
        K3 = GPy.kern.RBF(input_dim=self.input_dim)
        obj3_model = GPy.models.GPRegression(X, Y[1][:, np.newaxis], kernel=K3, normalizer=None)
        # X_vector = ndarray_to_vectors(self._get_var_name('search'), X)
        # X_design = [self._to_designspace(d) for d in X_vector]
        # X_nd = vectors_to_ndarray(self._get_var_name('design'), X_design)
        self.replace_feature = best_feature_by_gini(X, Y[0])
        X_nd = X.copy()
        X_nd[:, self.replace_feature] = np.clip(2 * (Y[1] - (-3)) / 6 - 1, -1, 1)
        K2 = GPy.kern.RBF(input_dim=self.input_dim)
        obj2_model = GPy.models.GPRegression(X_nd, Y[0][:, np.newaxis], kernel=K2, normalizer=None)
        obj2_model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-4)
        obj2_model['.*rbf.variance'].constrain_fixed(1.0)
        obj3_model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-4)
        obj3_model['.*rbf.variance'].constrain_fixed(1.0)

        self.model_list.append(obj3_model)
        self.model_list.append(obj2_model)

    def set_data(self, X, Y):
        self.model_list[0].set_XY(X, Y[1])
        self.replace_feature = best_feature_by_gini(X, Y[0])
        X_nd = X.copy()
        X_nd[:, self.replace_feature] = np.clip(2 * (Y[1] - (-3)) / 6 - 1, -1, 1)
        self.model_list[1].set_XY(X_nd, Y[1])


    def suggest(self, n_suggestions: Union[None, int] = None) -> List[Dict]:
        if self._X.size == 0:
            suggests = self.initial_sample()
            return suggests
        elif self._X.shape[0] < self.ini_num:
            pass
        else:
            if "normalize" in self.config:
                self.normalizer = get_normalizer(self.config["normalize"])

            Data = {"Target": {"X": self._X, "Y": self._Y}}
            self.update_model(Data)
            suggested_sample, acq_value = self.evaluator.compute_batch(
                None, context_manager=None
            )
            suggested_sample = self.search_space.zip_inputs(suggested_sample)
            suggested_sample = ndarray_to_vectors(
                self._get_var_name("search"), suggested_sample
            )
            design_suggested_sample = self.inverse_transform(suggested_sample)

            return design_suggested_sample

    def predict(self, X, full_cov=False):
        # X_copy = np.array([X])
        pred_mean = np.zeros((X.shape[0], 0))
        if full_cov:
            pred_var = np.zeros((0, X.shape[0], X.shape[0]))
        else:
            pred_var = np.zeros((X.shape[0], 0))

        Y_0, Y_0_var = self.model_list[0].predict(X, full_cov=full_cov)
        X_nd = X.copy()
        X_nd[:, self.replace_feature] = np.clip(2 * (Y_0 - (-3)) / 6 - 1, -1, 1)
        Y_1, Y_1_var = self.model_list[1].predict(X_nd)
        pred_mean = np.append(pred_mean, Y_0)
        pred_mean = np.append(pred_mean, Y_1)

        pred_var = np.append(pred_var, Y_0)
        pred_var = np.append(pred_var, Y_1)

        return pred_mean, pred_var


    def model_reset(self):
        self.model_list = []
        self.kernel_list = []

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self._X)

        return m.min()

    def get_fmin_by_id(self, idx):
        "Get the minimum of the current model."
        m, v = self.predict_by_id(self._X, idx)

        return m.min()
