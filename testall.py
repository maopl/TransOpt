import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from skimage import color
from collections import Counter
from transopt.benchmark.HPO import datasets
from transopt.benchmark.HPO.fast_data_loader import (FastDataLoader,
                                                     InfiniteDataLoader)
from transopt.benchmark.HPO.augmentation import mixup_data, mixup_criterion, AugMixDataset, ParameterizedAugmentation
import seaborn as sns

import os

# Create plots directory if it doesn't exist
plots_dir = './plots/image_analysis'
os.makedirs(plots_dir, exist_ok=True)

# 加载不同增强方式的数据集
augment_types = [None, 'cutout', 'autoaugment', 'mixup']
aug_datasets = {}
aug_images = {}
aug_labels = {}

for aug_type in augment_types:
    data_dir = None
    dataset = vars(datasets)['RobCifar10'](root=data_dir, augment=aug_type)
    aug_datasets[aug_type] = dataset
    
    # 提取图像和标签
    train_data = dataset.datasets['train']
    images = []
    labels = []
    for i in range(len(train_data)):
        image, label = train_data[i]
        images.append(image)
        labels.append(label)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    if aug_type == 'mixup':
        mixed_images, mixed_y_a, mixed_y_b, _ = mixup_data(images, labels, alpha=0.5, device="cpu")
        aug_images[aug_type] = mixed_images.numpy()
        aug_labels[aug_type] = mixed_y_a.numpy()
    else:
        aug_images[aug_type] = images.numpy()
        aug_labels[aug_type] = labels.numpy()

# 分析各增强方式的分布
def plot_distributions(aug_images, aug_labels):
    # 1. 类别分布
    plt.figure(figsize=(15, 5))
    for i, aug_type in enumerate(augment_types):
        plt.subplot(1, 4, i+1)
        class_counts = Counter(aug_labels[aug_type])
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.bar(classes, counts)
        plt.title(f'{aug_type} Class Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'augment_class_distributions.png'))
    plt.close()
    
    # 2. 颜色分布
    for channel, color_name in enumerate(['Red', 'Green', 'Blue']):
        plt.figure(figsize=(15, 5))
        for i, aug_type in enumerate(augment_types):
            plt.subplot(1, 4, i+1)
            channel_data = aug_images[aug_type][:, channel, :, :].flatten()
            plt.hist(channel_data, bins=50, color=color_name.lower(), alpha=0.7)
            plt.title(f'{aug_type} {color_name} Channel')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'augment_{color_name.lower()}_distributions.png'))
        plt.close()
    
    # 3. 纹理特征分布
    plt.figure(figsize=(15, 5))
    for i, aug_type in enumerate(augment_types):
        plt.subplot(1, 4, i+1)
        lbp_images = []
        for img in aug_images[aug_type][:1000]:
            gray = np.transpose(img, (1, 2, 0))
            gray = color.rgb2gray(gray)
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_images.append(lbp)
        lbp_images = np.array(lbp_images).flatten()
        plt.hist(lbp_images, bins=50, color='purple', alpha=0.7)
        plt.title(f'{aug_type} Texture Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'augment_texture_distributions.png'))
    plt.close()

# 计算分布之间的距离并绘制热力图
def compute_distribution_distances():
    # 创建距离矩阵
    distance_matrix = np.zeros((len(augment_types), len(augment_types)))
    
    for i, aug_type1 in enumerate(augment_types):
        for j, aug_type2 in enumerate(augment_types):
            # 使用KL散度或EMD计算分布距离
            hist1, _ = np.histogram(aug_images[aug_type1].flatten(), bins=50, density=True)
            hist2, _ = np.histogram(aug_images[aug_type2].flatten(), bins=50, density=True)
            
            # 使用欧氏距离作为简单的距离度量
            dist = np.sqrt(np.sum((hist1 - hist2) ** 2))
            distance_matrix[i, j] = dist
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, 
                xticklabels=augment_types,
                yticklabels=augment_types,
                annot=True,
                fmt='.4f',
                cmap='YlOrRd')
    plt.title('Distribution Distances Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distribution_distances_heatmap.png'))
    plt.close()
    
    # 打印距离矩阵
    print("\nDistribution Distances:")
    for i, aug_type1 in enumerate(augment_types):
        for j, aug_type2 in enumerate(augment_types):
            if i != j:
                print(f"{aug_type1}-{aug_type2}: {distance_matrix[i,j]:.4f}")




from scipy.stats import entropy
from scipy.stats import norm

def compute_gaussian_entropies():
    # 创建熵矩阵
    entropy_matrix = np.zeros((len(augment_types),))

    for i, aug_type in enumerate(augment_types):
        # Flatten the image data to fit a Gaussian distribution
        pixel_values = aug_images[aug_type].flatten()
        
        # Fit a Gaussian distribution
        mu, sigma = norm.fit(pixel_values)
        
        # Calculate the Shannon entropy of the Gaussian distribution
        ent = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
        entropy_matrix[i] = ent

    # 打印熵值
    print("\nGaussian Distribution Entropies:")
    for i, aug_type in enumerate(augment_types):
        print(f"{aug_type}: {entropy_matrix[i]:.4f}")

compute_gaussian_entropies()


def compute_distribution_entropies():
    # 创建熵矩阵
    entropy_matrix = np.zeros((len(augment_types),))

    for i, aug_type in enumerate(augment_types):
        # 计算每种增强方式的熵
        hist, _ = np.histogram(aug_images[aug_type].flatten(), bins=50, density=True)
        ent = entropy(hist)
        entropy_matrix[i] = ent

    # 打印熵值
    print("\nDistribution Entropies:")
    for i, aug_type in enumerate(augment_types):
        print(f"{aug_type}: {entropy_matrix[i]:.4f}")


plot_distributions(aug_images, aug_labels)
compute_distribution_entropies()
compute_distribution_distances()
