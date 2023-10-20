import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap(algorithms, test_problems):
    # 创建一个空的矩阵来存储结果
    results_matrix = np.empty((len(test_problems), len(algorithms)))

    # 逐个读取结果文件并填充矩阵
    for i, problem in enumerate(test_problems):
        for j, algorithm in enumerate(algorithms):
            # 从文件读取结果
            file_path = f"{algorithm}_{problem}.txt"
            with open(file_path, "r") as file:
                result = float(file.readline().strip())
            # 填充矩阵
            results_matrix[i, j] = result

    # 创建热力图
    fig, ax = plt.subplots()
    im = ax.imshow(results_matrix, cmap="viridis")

    # 设置轴标签
    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_yticks(np.arange(len(test_problems)))
    ax.set_xticklabels(algorithms)
    ax.set_yticklabels(test_problems)

    # 在热力图上显示数值
    for i in range(len(test_problems)):
        for j in range(len(algorithms)):
            text = ax.text(j, i, f"{results_matrix[i, j]:.2f}",
                           ha="center", va="center", color="w")

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 设置图形标题和标签
    plt.title("Algorithm Comparison Heatmap")
    plt.xlabel("Algorithms")
    plt.ylabel("Test Problems")

    # 显示图形
    plt.show()

# 示例用法

