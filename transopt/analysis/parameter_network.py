import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def calculate_importances(df, objective):
    """
    Calculates and returns parameter importances.
    """
    X = df.drop([objective], axis=1)
    y = df[objective]

    model = DecisionTreeRegressor()
    model.fit(np.array(X.values), np.array(y.values)[:, np.newaxis])
    feature_importances = model.feature_importances_

    parameter_importance_df = pd.DataFrame(
        {"Parameter": X.columns, "Importance": feature_importances}
    )
    return parameter_importance_df


def calculate_interaction(X, y):
    num_parameters = X.shape[1]
    h_matrix = np.zeros((num_parameters, num_parameters))

    # 训练单个变量的模型
    single_models = []
    for i in range(num_parameters):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X[:, [i]], y)
        single_models.append(model)

    # 两两特征组合，计算 H^2
    for (i, j) in combinations(range(num_parameters), 2):
        model_jk = RandomForestRegressor(n_estimators=50, random_state=42)
        model_jk.fit(X[:, [i, j]], y)
        f_jk = model_jk.predict(X[:, [i, j]])

        f_j = single_models[i].predict(X[:, [i]])
        f_k = single_models[j].predict(X[:, [j]])

        numerator = np.sqrt(np.sum((f_jk - f_j - f_k) ** 2))

        h_matrix[i, j] = numerator
        h_matrix[j, i] = h_matrix[i, j]

    h_matrix[h_matrix <= 0.01] = 0.01
    log_p = np.log(h_matrix)
    mean_log_p = np.mean(log_p)
    scaled_matrix = log_p - mean_log_p
    return scaled_matrix


def plot_network(data):
    G = nx.Graph()


if __name__ == "__main__":
    data = [{'x0': 2, 'x1': 4, 'x2': 2, 'x3': 2, 'x4': -3, 'f1': 51},
            {'x0': 3, 'x1': 3, 'x2': 3, 'x3': 3, 'x4': -2, 'f1': 52},
            {'x0': 4, 'x1': 2, 'x2': 4, 'x3': 4, 'x4': -1, 'f1': 53},
            {'x0': 5, 'x1': 1, 'x2': 5, 'x3': 5, 'x4': 0, 'f1': 54},
            {'x0': 6, 'x1': 0, 'x2': 6, 'x3': 6, 'x4': 1, 'f1': 55}]
    plot_network(data)