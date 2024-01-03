import heapq

import GPy
import numpy as np
import scipy.optimize as opt
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.util.general import get_quantiles
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.stats import *
from sklearn.metrics import pairwise_distances

from transopt.Optimizer.Acquisition.ACF import AcquisitionBase
from transopt.utils.hypervolume import calc_hypervolume
from transopt.utils.pareto import find_pareto_front
from transopt.utils.Register import acf_register


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
        self.children = {id} # 节点的唯一标识


def cluster_centroid(data, indices):
    # Assuming data is a 2D array-like structure and indices is a set of integer indices
    return np.mean([data[i] for i in indices], axis=0)


def verify_clusters(clusters):
    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters):
            if i != j and cluster_i['indices'].intersection(cluster_j['indices']):
                print('Cluster {} and {} are not disjoint'.format(i, j))
                return False
    return True

def hierarchical_clustering(data, nondominate_set_indices):
    # Convert dominate_set_indices to a set for efficient checking
    nondominate_set = set(nondominate_set_indices)

    clusters = [{'indices': {i}, 
                 'centroid': data[i], 
                 'dominant_count': 1 if i in nondominate_set else 0} 
                for i in range(len(data))]
    
    # Continue clustering until all clusters have a dominant element
    while not all(cluster['dominant_count'] > 0 for cluster in clusters):
        if len(clusters) == 1:  # Exit if only one cluster remains
            break

        # Calculate distance matrix
        centroid_matrix = np.array([cluster['centroid'] for cluster in clusters])
        distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

        merge_pairs = []  # Tracks pairs of clusters that will be merged
        merged_indices = set()  # Keep track of merged clusters
        for current_idx, current_cluster in enumerate(clusters):
            if current_cluster['dominant_count'] == 0 and current_idx not in merged_indices:
                closest_idx = np.argmin(distances[current_idx])
                if closest_idx not in merged_indices:
                    merge_pairs.append((current_idx, closest_idx))
                    merged_indices.update({current_idx, closest_idx})
        
        # Perform all merges
        for idx1, idx2 in merge_pairs:
            clusters[idx1]['indices'].update(clusters[idx2]['indices'])
            clusters[idx1]['centroid'] = cluster_centroid(data, clusters[idx1]['indices'])
            clusters[idx1]['dominant_count'] += clusters[idx2]['dominant_count']
        
        indices_to_remove = {idx2 for _, idx2 in merge_pairs}
        clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in indices_to_remove]
        
        # Check whether clusters
        total_indices = {i for cluster in clusters for i in cluster['indices']}
                
    return clusters

def clustering_by_nearest_nondominate(data, nondominate_set_indices):
    # Initialize clusters
    clusters = {idx: {'indices': set(), 'centroid': None} for idx in nondominate_set_indices}

    # Calculate the distance of each point to each nondominate point
    nondominate_points = [data[idx] for idx in nondominate_set_indices]
    all_distances = cdist(data, nondominate_points, metric='euclidean')
    
    # Assign each point to the cluster of its nearest nondominate point
    for point_idx, distances in enumerate(all_distances):
        nearest_nondominate_idx = nondominate_set_indices[np.argmin(distances)]
        clusters[nearest_nondominate_idx]['indices'].add(point_idx)
    
    # Calculate centroid for each cluster
    for idx, cluster in clusters.items():
        if cluster['indices']:
            cluster['centroid'] = np.mean([data[i] for i in cluster['indices']], axis=0)
        else:
            cluster['centroid'] = data[idx]  # Default to the nondominate point itself if no points are closer

    return clusters


def hierarchical_clustering_2order(data, clusters_ori, nondominate_set_indices):
    nondominate_set = set(nondominate_set_indices)

    # Initialize clusters and nodes
    clusters = [{'indices': c['indices'], 'centroid': c['centroid'],
                 'nondominated_set': {i for i in c['indices'] if i in nondominate_set}}
                for c in clusters_ori]
    
    nodes = [Node(c_id) for c_id in range(len(clusters))]
   
    # Merge clusters until all have non-dominant elements
    while not all(bool(cluster['nondominated_set']) for cluster in clusters):
        if len(clusters) == 1:  # Only one cluster left
            break

        centroid_matrix = np.array([cluster['centroid'] for cluster in clusters])
        distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        
        merge_pairs = []  # Tracks pairs of clusters that will be merged
        merged_indices = set()  # Keep track of merged clusters

        for current_idx, current_cluster in enumerate(clusters):
            if not bool(current_cluster['nondominated_set']) and current_idx not in merged_indices : # Only consider merging clusters without nondominated points
                closest_idx = np.argmin(distances[current_idx])
                if current_idx != closest_idx and closest_idx not in merged_indices:
                    merge_pairs.append((current_idx, closest_idx))
                    merged_indices.update({current_idx, closest_idx})
        
        # Perform all merges
        for idx1, idx2 in merge_pairs:
            clusters[idx1]['indices'].update(clusters[idx2]['indices'])
            clusters[idx1]['centroid'] = cluster_centroid(data, clusters[idx1]['indices'])
            clusters[idx1]['nondominated_set'].update(clusters[idx2]['nondominated_set'])
            nodes[idx1].children.update(nodes[idx2].children)
             
        indices_to_remove = {idx2 for _, idx2 in merge_pairs}
        clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in indices_to_remove]
        nodes = [node for k, node in enumerate(nodes) if k not in indices_to_remove]

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
        np.random.seed(0)
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



