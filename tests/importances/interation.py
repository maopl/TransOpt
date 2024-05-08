from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def logit_scale(matrix):
    log_p = np.log(matrix)  # 对每个元素计算对数
    mean_log_p = np.mean(log_p)  # 计算对数的平均值
    return log_p - mean_log_p  # 返回 g_k(x)

def calculate_interaction(X, y):
    n_features = X.shape[1]
    h_matrix = np.zeros((n_features, n_features))

    # 训练单个变量的模型
    single_models = []
    for i in range(n_features):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X[:, [i]], y)
        single_models.append(model)

    # 两两特征组合，计算 H^2
    for (i, j) in combinations(range(n_features), 2):
        # 包含两个变量的模型
        model_jk = RandomForestRegressor(n_estimators=50, random_state=42)
        model_jk.fit(X[:, [i, j]], y)
        f_jk = model_jk.predict(X[:, [i, j]])
        
        # 单个变量的预测
        f_j = single_models[i].predict(X[:, [i]])
        f_k = single_models[j].predict(X[:, [j]])

        # 计算 H^2
        numerator = np.sqrt(np.sum((f_jk - f_j - f_k) ** 2))

        h_matrix[i, j] = numerator
        h_matrix[j, i] = h_matrix[i, j]  # 由于 H^2 是对称的
    h_matrix[h_matrix <= 0.01] = 0.01
    scaled_matrix = logit_scale(h_matrix)
    
    return scaled_matrix

# 示例数据生成
np.random.seed(0)
X = np.random.normal(0, 1, (100, 5))  # 5个特征
y = 5 * X[:, 0] * X[:, 1] + 3 * X[:, 2] + X[:, 3] + np.random.normal(0, 0.5, 100)

# 计算 H^2 矩阵
h_squared_matrix = calculate_interaction(X, y)
print("H^2 matrix:\n", h_squared_matrix)
