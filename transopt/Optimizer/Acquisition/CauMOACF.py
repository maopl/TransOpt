import GPy
import numpy as np
import scipy.optimize as opt
from scipy.stats import *
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from GPyOpt.util.general import get_quantiles

from transopt.utils.Register import acf_register
from transopt.utils.hypervolume import calc_hypervolume
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from transopt.utils.pareto import find_pareto_front
from transopt.Optimizer.Acquisition.ACF import AcquisitionBase

import numpy as np
from sklearn.metrics import pairwise_distances


class Tree:
    def __init__(self, id, data=None):
        self.id = id
        self.data = data
        self.children = []

    def add_child(self, child):
        self.children.append(child)

class Node:
    def __init__(self, id):
        self.id = id
        self.children = [] # 节点的唯一标识


def cluster_centroid(data, indices):
    # Assuming data is a 2D array-like structure and indices is a set of integer indices
    return np.mean([data[i] for i in indices], axis=0)


def hierarchical_clustering(data, dominate_set_indices):
    # Convert dominate_set_indices to a set for efficient checking
    dominate_set = set(dominate_set_indices)

    # Initialize clusters
    clusters = [{'indices': {i}, 'centroid': data[i], 'is_non_dominated': i in dominate_set} for i in range(len(data))]

    centroid_matrix = np.array([cluster['centroid'] for cluster in clusters])
    distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
    np.fill_diagonal(distances, np.inf)

    # Merge clusters until all are non-dominated
    while any(cluster['is_non_dominated'] for cluster in clusters):
        if len(clusters) == 1:  # Only one cluster left
            break

        merged_indices = set()  # Keep track of merged clusters
        for i, cluster in enumerate(clusters) or cluster['is_non_dominated'] :
            if i in merged_indices:  # Skip already merged clusters
                continue

            # Find the nearest cluster to merge with
            j = np.argmin(distances[i])
            if j not in merged_indices and i != j:  # Ensure it's a valid cluster to merge
                # Merge clusters i and j
                cluster['indices'].update(clusters[j]['indices'])
                cluster['centroid'] = cluster_centroid(data, cluster['indices'])
                cluster['is_non_dominated'] = any(idx in dominate_set for idx in cluster['indices'])

                merged_indices.add(j)

        # Remove merged clusters and reconstruct the clusters list
        clusters = [cluster for k, cluster in enumerate(clusters) if k not in merged_indices]
        centroid_matrix = np.array([cluster['centroid'] for cluster in clusters])

        # Update distance matrix once after all merges
        distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

    return clusters


def hierarchical_clustering_2order(data, clusters_ori, dominate_set_indices):
    dominate_set = set(dominate_set_indices)
    mapping = {}
    # Initialize clusters
    clusters = []
    nodes = []
    for c_id, c in enumerate(clusters_ori):
        indices = c['indices']
        non_dominated_set = {}
        is_non_dominated = False
        for idx in indices:
            if idx in dominate_set_indices:
                is_non_dominated = True
                non_dominated_set.update(idx)

        clusters.append({'indices': indices, 'centroid':c['centroid'],
                         'is_non_dominated':is_non_dominated, 'non_dominated_set': non_dominated_set})
        nodes.append(Node(c_id))


    centroid_matrix = np.array([cluster['centroid'] for cluster in clusters])
    distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
    np.fill_diagonal(distances, np.inf)

    # Merge clusters until all are non-dominated
    while any(cluster['is_non_dominated'] for cluster in clusters):
        if len(clusters) == 1:  # Only one cluster left
            break

        merged_indices = set()  # Keep track of merged clusters
        for i, cluster in enumerate(clusters) or cluster['is_non_dominated']:
            if i in merged_indices:  # Skip already merged clusters
                continue

            # Find the nearest cluster to merge with
            j = np.argmin(distances[i])
            if j not in merged_indices and i != j:  # Ensure it's a valid cluster to merge
                # Merge clusters i and j
                cluster['indices'].update(clusters[j]['indices'])
                cluster['centroid'] = cluster_centroid(data, cluster['indices'])
                cluster['is_non_dominated'] = any(idx in dominate_set for idx in cluster['indices'])
                cluster['non_dominated_set'].update(clusters[j]['non_dominated_set'])
                nodes[i].children.append(i)
                nodes[i].children.append(j)
                merged_indices.add(j)

        # Remove merged clusters and reconstruct the clusters list
        clusters = [cluster for k, cluster in enumerate(clusters) if k not in merged_indices]
        nodes = [nodes[k] for k, cluster in enumerate(clusters) if k not in merged_indices]
        centroid_matrix = np.array([cluster['centroid'] for cluster in clusters])

        # Update distance matrix once after all merges
        distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

    return clusters, nodes


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
        pareto, pareto_index = find_pareto_front(Mean, return_index=True)
        # if len(pare_index) == 1:
        #     return x[pare_index]
        # else:
        #     clusters = hierarchical_clustering(Mean, pare_index)

        clusters = hierarchical_clustering(Mean, pareto_index)
        pareto_second_order, pareto_second_index = find_pareto_front(Mean[:, 1:], return_index=True)
        new_clusters, relation= hierarchical_clustering_2order(Mean, clusters, pareto_second_index)
        root = Tree(id=0)
        [root.add_child(Tree(k)) for k, c in enumerate(new_clusters)]
        for child in root.children:
            [child.add_child(Tree(k)) for k in relation[child.id]]


        return



