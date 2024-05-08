from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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

    mean = np.mean(h_matrix)
    std = np.std(h_matrix)

    normalized_matrix = (h_matrix - mean) / std
    scaled_matrix = 1 / (1 + np.exp(-normalized_matrix))
    
    return scaled_matrix

def calculate_importances(df, objective):
    """
    Calculates and returns feature importances.
    """
    X = df.drop([objective], axis=1)
    y = df[objective]

    model = DecisionTreeRegressor()
    model.fit(np.array(X.values), np.array(y.values)[:, np.newaxis])
    feature_importances = model.feature_importances_

    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances}
    )
    return feature_importance_df



    
    
G = nx.Graph()

# 添加节点及其权重，权重范围在 0 到 1
nodes = ["Books", "Personal", "F.Undergrad", "Apps", "Accept", "Private", "P.Undergrad", "PhD", "S.F.Ratio", "Outstate", "Room.Board"]
node_weights = np.random.uniform(0, 1, size=len(nodes))  # 生成0到1之间的随机权重
for node, weight in zip(nodes, node_weights):
    G.add_node(node, weight=weight)

# 为每对节点添加边，边的权重也在 0 到 1 之间
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        weight = np.random.uniform(0, 1)  # 生成一个0到1之间的随机权重
        G.add_edge(nodes[i], nodes[j], weight=weight)

# 设置节点的位置为圆形布局
pos = nx.circular_layout(G)

# 创建颜色映射
node_cmap = plt.cm.Greens
edge_cmap = plt.cm.Blues

# 节点的颜色根据权重映射
node_color = [node_cmap(data['weight']) for v, data in G.nodes(data=True)]
node_size = [data['weight'] * 1000 + 100 for v, data in G.nodes(data=True)]
node_alpha = [data['weight'] for v, data in G.nodes(data=True)]  # 透明度根据权重调整

# 绘制网络图
edges = G.edges(data=True)
nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, alpha=node_alpha)
nx.draw_networkx_labels(G, pos)

# 单独绘制每条边，设置颜色和透明度
for u, v, data in edges:
    color = edge_cmap(data['weight'])
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1, alpha=data['weight'], edge_color=[color])

# 显示图形
plt.show()
