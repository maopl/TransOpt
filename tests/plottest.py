import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 创建图对象
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
