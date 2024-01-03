import GPy
import numpy as np
import scipy.optimize as opt
from scipy.stats import *
from scipy.spatial import distance
from GPyOpt.util.general import get_quantiles

from transopt.utils.Register import acf_register
from transopt.utils.hypervolume import calc_hypervolume
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from transopt.utils.pareto import find_pareto_front
from transopt.Optimizer.Acquisition.ACF import AcquisitionBase

import numpy as np
from sklearn.metrics import pairwise_distances

class ClusterNode:
    def __init__(self, members, centroid):
        self.members = members  # 簇成员索引
        self.children = []  # 子簇
        self.centroid = centroid  # 簇的质心

def cluster_centroid(data, cluster):
    # 计算给定簇的质心
    cluster_data = data[list(cluster)]
    return np.mean(cluster_data, axis=0)

def hierarchical_clustering(data, dominate_set_indices):
    # Convert dominate_set_indices to a set for efficient checking
    dominate_set = set(dominate_set_indices)

    # Initialize clusters and centroids
    clusters = [{i} for i in range(len(data))]
    centroids = [data[i] for i in range(len(data))]

    # Initialize a list to track if a cluster has a non-dominated solution
    contains_dominated = [not (cluster & dominate_set) for cluster in clusters]

    while any(contains_dominated):  # Continue until all clusters have a non-dominated solution
        merged_in_this_pass = set()  # Keep track of merged clusters to avoid double merging

        for i in range(len(clusters) - 1, -1, -1):
            if i in merged_in_this_pass:  # Skip if this cluster has already been merged in this pass
                continue

            if contains_dominated[i]:
                # Calculate distances between this centroid and others
                distances_to_others = pairwise_distances([centroids[i]], centroids, metric='euclidean')[0]
                distances_to_others[i] = np.inf  # Ignore distance to itself
                for idx in merged_in_this_pass:  # Avoid merging with clusters already merged in this pass
                    distances_to_others[idx] = np.inf

                # Find the closest cluster to merge with
                closest_cluster_index = np.argmin(distances_to_others)

                # Merge clusters
                clusters[i].update(clusters[closest_cluster_index])
                centroids[i] = cluster_centroid(data, clusters[i])

                # Update the 'contains_dominated' list for the new cluster
                contains_dominated[i] = not (clusters[i] & dominate_set)

                # Mark the merged clusters
                merged_in_this_pass.add(closest_cluster_index)
                merged_in_this_pass.add(i)

                # Remove the merged cluster
                del clusters[closest_cluster_index]
                del centroids[closest_cluster_index]
                del contains_dominated[closest_cluster_index]

        # Recalculate contains_dominated for remaining clusters
        contains_dominated = [not (cluster & dominate_set) for cluster in clusters]

    return clusters


@acf_register("CauMOACF")
class CauMOACF:

    def __init__(self, model, space, optimizer, config):
        self.optimizer = optimizer
        self.model = model
        self.model_id = 0
        if 'jitter' in config:
            self.jitter = config['jitter']
        else:
            self.jitter = 0.1

        if 'threshold' in config:
            self.threshold = config['threshold']
        else:
            self.threshold = 0

        self.members = None
        self.centroid = None



    def optimize(self, duplicate_manager=None):
        x = np.random.random(size=(10000, self.model.input_dim)) *2 - 1
        Mean, Var = self.model.predict(x)
        pare, pare_index = find_pareto_front(Mean, return_index=True)
        clusters = hierarchical_clustering(Mean, pare_index)


        return



