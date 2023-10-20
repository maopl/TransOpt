import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn import neural_network

# datasets
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits
from sklearn.datasets import fetch_covtype, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import numpy as np

import sys
sys.path.append("..")
from data_environment import SimEnvironment, SimEnvironmentClassification
import prapare_dataset
import tensorflow as tf
from trieste.data import Dataset
from trieste.space import Box
import xgboost as xgb

import pickle
import sys
import matplotlib.pyplot as plt


class Sim(SimEnvironment):
    def __init__(self, threshold=0., real_value=True, virtual_value=2., dataset='cancer', method='rf', memory_threshold=1e5):
        self._method = method
        self._dataset = dataset
        self._memory_threshold = memory_threshold
        super().__init__(threshold, real_value, virtual_value)

    def observer(self, query_point):
        # define the objective
        memory_test = Memory(self._dataset)
        query_point_np = query_point.numpy()

        accuracy = np.zeros(len(query_point_np[:, 0]))
        memory_cons = np.zeros(len(query_point_np[:, 0]))
        for i in range(len(query_point_np[:, 0])):
            if self._method == 'rf':
                accuracy[i], memory_cons[i] = memory_test.rf_evaluation(query_point_np[i, :])
            elif self._method == 'xgb':
                accuracy[i], memory_cons[i] = memory_test.xgboost_evaluation(query_point_np[i, :])
            elif self._method == 'svm':
                accuracy[i], memory_cons[i] = memory_test.svm_evaluation(query_point_np[i, :])
            elif self._method == 'nn':
                accuracy[i], memory_cons[i] = memory_test.nn_evaluation(query_point_np[i, :])

        # define the constraints
        y = (memory_cons - self._memory_threshold) / self._memory_threshold
        feasible_query_point = query_point[y <= 0]
        obj = accuracy[y <= 0][:, None]

        noise_var = memory_cons * 0.0 + 1e-6
        for i in range(len(y)):
            if y[i] > 0:
                # generate the real data output with small noise of failure point
                y[i] = np.log(1 + self._virtual_value)
                noise_var[i] = (0.5 * np.abs(y[i])) ** 2 + 1e-6
            else:
                # if not observed, generate the virtual value of 2.
                # y[i] = -virtual_value
                # noise_var[i] = (0.5 * np.abs(y[i])) ** 2 + 1e-6

                # if observed with certain message, it is suggested to use the projection function $- log(1 - x)$ to
                # boost changes at the boundary
                y[i] = - np.log(- y[i] / 40000 + 1)
        y = tf.convert_to_tensor(y.reshape(-1, 1), query_point.dtype)
        noise_var = tf.convert_to_tensor(noise_var.reshape(-1, 1), query_point.dtype)
        cons = tf.concat([y, noise_var], 1)
        print("query points = ", query_point)
        print("searching obj = ", obj)
        print("searching cons = ", y)
        return {
            self.OBJECTIVE: Dataset(feasible_query_point, obj),
            self.CONSTRAINT: Dataset(query_point, cons),
        }


class Sim_binary(SimEnvironmentClassification):
    def __init__(self, threshold=0., dataset='cancer', method='rf', memory_threshold=1e5, feasible_value=1., infeasible_value=0.):
        self._method = method
        self._dataset = dataset
        self._memory_threshold = memory_threshold
        # self._feasible_value = 1.
        # self._infeasible_value = -1.
        super().__init__(threshold, feasible_value=feasible_value, infeasible_value=infeasible_value)

    def observer(self, query_point):
        # define the objective
        memory_test = Memory(self._dataset)
        query_point_np = query_point.numpy()

        accuracy = np.zeros(len(query_point_np[:, 0]))
        memory_cons = np.zeros(len(query_point_np[:, 0]))
        for i in range(len(query_point_np[:, 0])):
            if self._method == 'rf':
                accuracy[i], memory_cons[i] = memory_test.rf_evaluation(query_point_np[i, :])
            elif self._method == 'xgb':
                accuracy[i], memory_cons[i] = memory_test.xgboost_evaluation(query_point_np[i, :])
            elif self._method == 'svm':
                accuracy[i], memory_cons[i] = memory_test.svm_evaluation(query_point_np[i, :])
            elif self._method == 'nn':
                accuracy[i], memory_cons[i] = memory_test.nn_evaluation(query_point_np[i, :])

        # define the constraints
        y = memory_cons - self._memory_threshold
        feasible_query_point = query_point[y <= 0]
        obj = accuracy[y <= 0][:, None]

        for i in range(len(y)):
            if y[i] > 0:
                # generate the real data output with small noise of failure point
                y[i] = self._infeasible_value
                # noise_var[i] = (0.5 * np.abs(y[i])) ** 2 + 1e-6
            else:
                # if not observed, generate the virtual value of 2.
                # y[i] = -virtual_value
                # noise_var[i] = (0.5 * np.abs(y[i])) ** 2 + 1e-6

                # if observed with certain message, it is suggested to use the projection function $- log(1 - x)$ to
                # boost changes at the boundary
                y[i] = self._feasible_value
        cons = tf.convert_to_tensor(y.reshape(-1, 1), query_point.dtype)
        print("query points = ", query_point)
        print("searching obj = ", obj)
        print("searching cons = ", memory_cons - self._memory_threshold)
        return {
            self.OBJECTIVE: Dataset(feasible_query_point, obj),
            self.CONSTRAINT: Dataset(query_point, cons),
        }

class Memory:
    def __init__(self, dataset='cancer'):

        # self.dataset = prapare_dataset.dataset_german()
        X_train = None
        y_train = None
        X_test = None
        y_test = None
        self.classification = True
        if dataset == 'cancer':
            X, y = load_breast_cancer(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'iris':
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'digit':
            X, y = load_digits(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'diabetes':
            self.classification = False
            X, y = load_diabetes(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'cover':
            X, y = fetch_covtype(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'house':
            self.classification = False
            X, y = fetch_california_housing(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        else:
            print("invalid dataset.")
            exit()

        self.dataset = {
            "trainX": X_train,
            "trainY": y_train,
            "testX": X_test,
            "testY": y_test,
        }

    def rf_evaluation(self, hyperparameters):
        # max_depth: 1-50 int, log
        # min_samples_split 2 - 128 int, log
        # n_estimators 0-64 int, log
        # min_samples_leaf 0-20 int
        # max_features (0,1) float
        max_depth = np.floor(np.power(10, hyperparameters[0])).astype(int)
        min_samples_split = np.floor(np.power(2, hyperparameters[1])).astype(int)
        n_estimators = np.floor(np.power(10, hyperparameters[2])).astype(int)
        min_samples_leaf = np.floor(hyperparameters[3]).astype(int)
        max_features = hyperparameters[4]
        if self.classification:
            classifier = RandomForestClassifier(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                max_features=max_features,
                                                min_samples_leaf=min_samples_leaf,
                                                # criterion=criterion,
                                                n_estimators=n_estimators,
                                                )

            classifier.fit(self.dataset['trainX'], self.dataset['trainY'])
            p = pickle.dumps(classifier)
            memory_cost = sys.getsizeof(p)
            pred = classifier.predict(self.dataset['testX'])

            accuracy = accuracy_score(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return -accuracy, memory_cost
        else:
            regressor = RandomForestRegressor(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                max_features=max_features,
                                                min_samples_leaf=min_samples_leaf,
                                                # criterion=criterion,
                                                n_estimators=n_estimators,
                                                )

            regressor.fit(self.dataset['trainX'], self.dataset['trainY'])
            p = pickle.dumps(regressor)
            memory_cost = sys.getsizeof(p)
            pred = regressor.predict(self.dataset['testX'])

            accuracy = mean_squared_error(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return accuracy, memory_cost


    def xgboost_evaluation(self, hyperparameters):
        # eta: (2^**-10)-1. log, float
        # max_depth: 1-15   int
        # colsample_bytree 0.01-1. float
        # lambda_reg (2^**-10)-2**10 log, float
        # alpha_reg (2^**-10)-2**10 log, float
        # min_child_weight 1-2**7 log, float
        # n_estimator 1-2^**8 log, int
        eta = np.power(2, hyperparameters[0])
        max_depth = np.floor(hyperparameters[1]).astype(int)
        colsample_bytree = hyperparameters[2]
        lambda_reg = np.power(2, hyperparameters[3])
        alpha_reg = np.power(2, hyperparameters[4])
        min_child_weight = np.power(2, hyperparameters[5])
        n_estimator = np.floor(np.power(2, hyperparameters[6])).astype(int)

        if self.classification:
            classifier = xgb.XGBClassifier(
                learning_rate=eta,
                n_estimators=n_estimator,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                colsample_bytree=colsample_bytree,
                reg_lambda=lambda_reg,
                reg_alpha=alpha_reg,
                random_state=1,
            )
            classifier.fit(self.dataset['trainX'], self.dataset['trainY'])
            pred = classifier.predict(self.dataset['testX'])
            p = pickle.dumps(classifier)
            memory_cost = sys.getsizeof(p)

            accuracy = accuracy_score(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return -accuracy, memory_cost
        else:
            regressor = xgb.XGBRegressor(
                learning_rate=eta,
                n_estimators=n_estimator,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                colsample_bytree=colsample_bytree,
                reg_lambda=lambda_reg,
                reg_alpha=alpha_reg,
                random_state=1,
            )
            regressor.fit(self.dataset['trainX'], self.dataset['trainY'])
            pred = regressor.predict(self.dataset['testX'])
            p = pickle.dumps(regressor)
            memory_cost = sys.getsizeof(p)

            accuracy = mean_squared_error(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return accuracy, memory_cost

    def svm_evaluation(self, hyperparameters):
        gamma = np.power(2, hyperparameters[0])
        C = np.power(2, hyperparameters[1])
        # gamma = hyperparameters[0]
        # C = hyperparameters[1]
        classifier = svm.SVC(gamma=gamma, C=C,
                             )
        # p = pickle.dumps(classifier)
        # print("the model size after training is: ", sys.getsizeof(p))

        classifier.fit(self.dataset['trainX'], self.dataset['trainY'])
        p = pickle.dumps(classifier)


        pred = classifier.predict(self.dataset['testX'])
        memory_cost = sys.getsizeof(p)

        accuracy = accuracy_score(pred, self.dataset['testY'])
        print("accuracy: ", accuracy, "     model size: ", memory_cost)
        return -accuracy, memory_cost


    def nn_evaluation(self, hyperparameters):
        # hidden_layer_sizes (a, b): (2^2)-(2^8). log, int
        # batch_size 2^2-2^8 log, int
        # alpha (10^-8)-(10^-3) log, float
        # learning_rate_init (10^-5)-(10^0), log, float
        # tol (10^-6)-(10^-2) log, float
        # beta_1 0.-0.9999 float
        # beta_2 0.-0.9999 float
        hidden_layer_sizes_a = np.round(np.power(2, hyperparameters[0])).astype(int)
        hidden_layer_sizes_b = np.round(np.power(2, hyperparameters[1])).astype(int)
        batch_size = np.round(np.power(2, hyperparameters[2])).astype(int)
        alpha = np.power(10, hyperparameters[3])
        learning_rate_init = np.power(10, hyperparameters[4])
        tol = np.power(10, hyperparameters[5])
        beta_1 = hyperparameters[6]
        beta_2 = hyperparameters[7]

        classifier = neural_network.MLPClassifier(
            hidden_layer_sizes=(hidden_layer_sizes_a, hidden_layer_sizes_b),
            batch_size=batch_size,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            tol=tol,
            beta_1=beta_1,
            beta_2=beta_2,
            random_state=1,
        )
        classifier.fit(self.dataset['trainX'], self.dataset['trainY'])
        p = pickle.dumps(classifier)
        pred = classifier.predict(self.dataset['testX'])

        memory_cost = sys.getsizeof(p)
        accuracy = accuracy_score(pred, self.dataset['testY'])
        print("accuracy: ", accuracy, "     model size: ", memory_cost)
        return -accuracy, memory_cost


def mesh_svm():
    x1 = np.linspace(-10, 0, 10)
    x2 = np.linspace(-2, 4, 10)
    x1 = np.power(10, x1)
    x2 = np.power(10, x2)
    mesh_x1, mesh_x2 = np.meshgrid(x1, x2)
    a1 = mesh_x1.reshape(-1, 1)
    a2 = mesh_x2.reshape(-1, 1)
    config = np.hstack((a1, a2))
    return mesh_x1, mesh_x2, config


if __name__ == '__main__':
    memory_test = Memory('cover')

    '''
    # svm test
    grid_X, grid_Y, parameters = mesh_svm()
    acc = parameters[:, 0] * 0.
    mem = parameters[:, 0] * 0.
    for i in range(len(parameters[:, 0])):
        acc[i], mem[i] = memory_test.svm_evaluation(parameters[i, :])
    acc_mean = acc.reshape(grid_X.shape)
    dsp_mean = mem.reshape(grid_X.shape)
    fig = plt.figure()
    plt.semilogx()
    plt.semilogy()
    c = plt.contourf(grid_X, grid_Y, acc_mean)
    plt.colorbar(c)

    fig = plt.figure()
    plt.semilogx()
    plt.semilogy()
    c = plt.contourf(grid_X, grid_Y, dsp_mean)
    plt.colorbar(c)

    plt.show()
    '''
    size = 2000

    # # rf test
    # # np.([np.log10(1), np.log2(2), np.log10(1), 1, 0.],
    # #  [np.log10(50), np.log2(128), np.log10(100), 21 - 1e-10, 1.])
    # x1 = np.random.uniform(np.log10(1), np.log10(50), size)[:, None]
    # x2 = np.random.uniform(np.log2(2), np.log2(128), size)[:, None]
    # x3 = np.random.uniform(np.log10(1), np.log10(100), size)[:, None]
    # x4 = np.random.uniform(1, 21-1e-10, size)[:, None]
    # x5 = np.random.uniform(0, 1., size)[:, None]


    # xgboost test
    #np.([-10, 1, 0.01, -10, -10, 0, 0],
    #    [0, 15, 1., 10, 10, 7, 8])
    x1 = np.random.uniform(-10, 0, size)[:, None]
    x2 = np.random.uniform(1, 15, size)[:, None]
    x3 = np.random.uniform(0.01, 1., size)[:, None]
    x4 = np.random.uniform(-10, 10, size)[:, None]
    x5 = np.random.uniform(-10, 10, size)[:, None]
    x6 = np.random.uniform(0, 7, size)[:, None]
    x7 = np.random.uniform(0, 8, size)[:, None]


    parameters = np.hstack((x1, x2, x3, x4, x5, x6, x7))

    acc = parameters[:, 0] * 0.
    mem = parameters[:, 0] * 0.

    for i in range(len(parameters[:, 0])):
        acc[i], mem[i] = memory_test.xgboost_evaluation(parameters[i, :])

    print("accuracy: ", np.sort(acc))
    # print("memory cost: ", mem)
    print("quantile0.3: ", np.quantile(mem, 0.3), "\tquantile0.5: ", np.quantile(mem, 0.5), "\tquantile0.7: ", np.quantile(mem, 0.7))