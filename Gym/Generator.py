import matplotlib.pyplot as plt
import numpy as np
# from transopt.Benchmark.Synthetic.SyntheticBenchmark import Select_synthetic_fun
from transopt.benchmark.RL.LunarlanderBenchmark import LunarlanderBenchmark
from sklearn.preprocessing import power_transform
from transopt.utils.Normalization import Normalize
from transopt.benchmark.HPOB.HpobBench import HPOb
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from transopt.utils.Normalization import normalize


class PseudoPointsGenerator():
    def __init__(self, inner_radius=1, outer_radius=20, n_components = 5):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', reg_covar=0.1)


    def bandpass_filter_2D(self, shape, inner_radius, outer_radius):
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        rows_range, cols_range = np.ogrid[:rows, :cols]
        distance_from_center = np.sqrt((rows_range - center_row)**2 + (cols_range - center_col)**2)
        mask = np.logical_and(distance_from_center >= inner_radius, distance_from_center <= outer_radius)
        return mask.astype(float)

    def generate_from_fft(self, X, Y, target_samples=1000):
        # 对Y进行排序并获取相应的X序列
        sorted_indices = np.argsort(Y)
        sorted_X = X[sorted_indices]

        # 对X进行2D傅里叶变换，并将低频成分移至中心
        X_freq = np.fft.fftshift(np.fft.fft2(sorted_X), axes=0)

        N, M = X.shape
        if N > target_samples:
            raise ValueError("Target length should be greater than the original length.")

        # 创建新的频域数据并初始化为0
        expanded_freq_data = np.zeros((target_samples, M), dtype=np.complex128)

        # 定位低频和高频的起始点
        start_N = (target_samples - N) // 2

        # 插入原始的频域数据
        expanded_freq_data[start_N:start_N+N, :] = X_freq

        # bp_filter = self.bandpass_filter_2D(expanded_freq_data.shape, self.inner_radius, self.outer_radius)
        # expanded_freq_data = expanded_freq_data * bp_filter

        # 进行逆傅里叶变换并将频率数据移回原始位置
        expanded_X = np.fft.ifft2(np.fft.ifftshift(expanded_freq_data, axes=0)).real

        # expanded_X = expanded_X * (target_samples / N)

        max_values = np.max(expanded_X, axis=0)
        min_values = np.min(expanded_X, axis=0)

        # 计算每列的范围
        ranges = max_values - min_values

        # 归一化到0-1范围
        normalized_matrix = (expanded_X - min_values) / ranges

        # 缩放到-1到1范围
        normalized_matrix = 2 * normalized_matrix - 1

        return normalized_matrix[::-1]


    def Set_Y_by_gaussian(self, X, Y, test_X):

        conca_X = np.concatenate((X, test_X))
        self.gmm.fit(conca_X)
        prob = self.gmm.predict_proba(conca_X)

        N, M = prob.shape
        new_P = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                new_value = prob[i, j]
                for k in range(M):
                    if k != j:
                        new_value *= (1 - prob[i, k])
                new_P[i, j] = new_value

        label = np.argmax(new_P, axis=1)

        mean_value = []
        # set mean value for each component
        for i in range(self.n_components):
            indices = np.where(label[:len(X)] == i)[0]
            if len(indices) == 0:
                mean_value.append(np.mean(Y))
                continue
            # 获取对应的 Y 值和 new_P
            Y_values = Y[indices]
            new_P_values = new_P[indices, i]
            # 只考虑概率大于 0.5 的值
            valid_indices = new_P_values > 0.2
            valid_Y_values = Y_values[valid_indices]
            valid_new_P_values = new_P_values[valid_indices]
            # 计算加权平均值
            mean_value.append(np.sum(valid_Y_values * valid_new_P_values) / np.sum(valid_new_P_values))

        mean_value = np.array(mean_value)
        test_Y = []
        for i in range(len(test_X)):
            # 获取 test_X[i] 的 label
            p = new_P[i]
            # 获取所有 X 向量中 label 为 lbl 的向量的索引

            test_Y.append(np.sum(mean_value * p) / np.sum(p))

        test_Y = np.array(test_Y)

        # # 检测数组中是否存在NaN值
        # has_nan = np.isnan(test_Y).any()
        #
        # if has_nan:
        #     print("数组中存在NaN值")
        # else:
        #     print("数组中没有NaN值")
        mean_Y = np.mean(test_Y)
        std_Y = np.std(test_Y)




        return Norm_pt(test_Y[:,np.newaxis], mean_Y, std_Y)

    def Set_Y_by_KDE(self, X, Y, test_X):
        test_Y = []
        # 创建并训练KMeans模型
        kmeans = KMeans(n_clusters=self.n_components, random_state=0).fit(X)
        # 将每个簇的数据保存在一个列表中
        clusters = [X[kmeans.labels_ == i] for i in range(self.n_components)]
        cluster_Y = [Y[kmeans.labels_ == i] for i in range(self.n_components)]
        # 对每个簇的数据拟合KDE
        kde_models = [KernelDensity().fit(cluster) for cluster in clusters]

        predicted_cluster = kmeans.predict(test_X)

        for x_id, c in enumerate(predicted_cluster):
            # 使用KDE模型为该簇的X值计算权重
            weights = np.exp(kde_models[c].score_samples([test_X[x_id]]))

            # 使用这些权重和该簇的Y值计算加权平均值
            test_Y.append(np.sum(weights * cluster_Y) / np.sum(weights))
        cluster_Y_values = Y[cluster_indices]

        return np.array(test_Y)
    


if __name__ == '__main__':
    # 示例
    lunar_seeds = [2, 3, 4, 5, 10, 14, 15, 19]
    dim = 10
    G = PseudoPointsGenerator()

    # problem = HPOb(search_space_id='5965', data_set_id='9946',xdim=10, path='../Benchmark/HPOB/hpob-data/')
    # indexs = np.random.choice(range(1,13000), size=60)
    # X_original = problem.get_var(indexs=indexs)
    # Y = problem.f(X=None, indexs=indexs)

    # problem = Lunar_lander(name=f'Lunar_lander_{dim}d', Xdim=dim, Seed=2)
    problem = Select_synthetic_fun(fun_name='Sphere', input_dim=dim, Seed=0, dtype=np.float)
    bounds = problem.bounds
    np.random.seed(38)
    X_original = np.random.uniform(-1,1,size=(6*dim,dim))
    # a = np.random.random(size=(3*dim,dim)) * 0.4 + 0.2
    # X_original = np.concatenate((X_original,a))
    Y = problem.f(X_original)

    X_generated = G.generate_from_fft(X_original, Y, target_samples=2000)
    new_Y = problem.f(X_generated)
    pseudo_Y = G.Set_Y_by_gaussian(X_original, Y, test_X=X_generated)

    # 绘制原始X矩阵的热图
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.imshow(X_original, cmap='viridis', aspect='auto')
    plt.title('Original X Heatmap')

    # 绘制生成的X矩阵的热图
    plt.subplot(2, 2, 2)
    plt.imshow(X_generated, cmap='viridis', aspect='auto')
    plt.title('Generated X Heatmap')


    # 绘制Y值的一维曲线
    sorted_indices = np.argsort(Y)
    plt.subplot(2, 2, 3)
    rank = np.argsort(np.argsort(pseudo_Y, axis=0), axis=0)
    new_pseudo_vector = rank + 1

    plt.plot(range(len(X_generated)), new_pseudo_vector, label='Generated X')
    rank = np.argsort(np.argsort(new_Y, axis=0), axis=0)
    new_vector = rank + 1
    plt.plot(range(len(X_generated)), new_vector, label='Real rank')
    plt.xlabel('Y')
    plt.ylabel('Mean of Generated X')
    plt.title('Generated sorted Y vs Real sorted Y')
    plt.legend()

    # 计算Pearson相关系数
    pearson_corr, _ = pearsonr(new_pseudo_vector, new_vector)
    print('Pearson Correlation:', pearson_corr)

    # 计算Spearman相关系数
    spearman_corr, _ = spearmanr(pseudo_Y, new_Y)
    print('Spearman Correlation:', spearman_corr)

    plt.subplot(2, 2, 4)
    # 添加文本到图中
    plt.text(0.1, 0.9, 'Pearson Correlation: {:.2f}'.format(pearson_corr), fontsize=9, transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, 'Spearman Correlation: {:.2f}'.format(spearman_corr), fontsize=9, transform=plt.gca().transAxes)

    plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.savefig(f'Sphere')
