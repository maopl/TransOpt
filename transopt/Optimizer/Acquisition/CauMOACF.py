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
from transopt.utils.Normalization import normalize

class Tree:
    def __init__(self, secondary_nodes, evaluated_points):
        self.secondary_nodes = secondary_nodes
        self.evaluated_points = evaluated_points
        
        self.attach_evaluated_points_to_nodes()
        self.cal_ucb_for_all()

    def attach_evaluated_points_to_nodes(self):
        # Gather all centroids
        secondary_centroids = [node.centroid for node in self.secondary_nodes]
        primary_nodes = [node for sec_node in self.secondary_nodes for node in sec_node.children]
        primary_centroids = [node.centroid for node in primary_nodes]
        
        # Calculate distances and attach points for secondary nodes
        distances_to_secondary = cdist(self.evaluated_points, secondary_centroids, metric='euclidean')
        closest_secondary_idx = np.argmin(distances_to_secondary, axis=1)
        for idx, sec_idx in enumerate(closest_secondary_idx):
            self.secondary_nodes[sec_idx].evaluated_points.append(self.evaluated_points[idx])
        
        # Calculate distances and attach points for primary nodes
        distances_to_primary = cdist(self.evaluated_points, primary_centroids, metric='euclidean')
        closest_primary_idx = np.argmin(distances_to_primary, axis=1)
        for idx, pri_idx in enumerate(closest_primary_idx):
            primary_nodes[pri_idx].evaluated_points.append(self.evaluated_points[idx])
            
    def cal_ucb_for_all(self):
        for sec_node in self.secondary_nodes:
            sec_node.ucb = self.cal_ucb(sec_node.nondominants_2, sec_node.evaluated_points, len(self.evaluated_points))
            
            for pri_node in sec_node.children:
                pri_node.ucb = self.cal_ucb(pri_node.nondominants_3, pri_node.evaluated_points, len(sec_node.evaluated_points))
    
    def cal_ucb(self, nondominants, evaluated_points, parent_evaluated_count):
        n_pf = len(nondominants)
        n_evaluated = len(evaluated_points)
        n_evaluated = 0.1 if n_evaluated == 0 else n_evaluated  # Avoid division by zero
        
        return n_pf / n_evaluated + 0.1 * np.sqrt(parent_evaluated_count / n_evaluated)
        
    def get_candidate(self):
        # Find the secondary node with the highest UCB
        sec_node = max(self.secondary_nodes, key=lambda node: node.ucb)
        # Find the primary node with the highest UCB within the selected secondary node
        pri_node = max(sec_node.children, key=lambda node: node.ucb)
        
        return pri_node

class ClusterTreeNode:
    def __init__(self, members, centroid, nondominants_3, nondominants_2, children=None):
        self.members = members  # All members of this cluster
        self.centroid = centroid
        self.nondominants_3 = nondominants_3  # Nondominants based on 3 objectives
        self.nondominants_2 = nondominants_2  # Nondominants based on 2 objectives
        self.children = children if children is not None else []  # Children of this node (subclusters)
        
        self.evaluated_points = [] # Evaluated points attached to this node
        self.ucb = None  # UCB value for this node    
    
        
class NondominatedCluster:
    def __init__(self, dataset, cluster_members, nondominated_indices=set()):
        self.members = set(cluster_members)  # Members of this cluster
        self.nondominants = set(nondominated_indices)  # Nondominated points in this cluster
        self.centroid = self.calculate_centroid(dataset)  # Centroid of the cluster

    def merge_clusters(self, other_cluster, dataset):
        self.members.update(other_cluster.members)
        self.nondominants.update(other_cluster.nondominants)
        self.centroid = self.calculate_centroid(dataset)

    def calculate_centroid(self, dataset):
        return np.mean([dataset[index] for index in self.members], axis=0)


class AggregatedCluster:
    def __init__(self, dataset, subclusters, higher_nondominated_indices=set()):
        self.subclusters = set(subclusters)  # Subclusters aggregated into this cluster
        self.members = set.union(*(sc.members for sc in subclusters))  # Members from all subclusters
        self.higher_nondominants = set(higher_nondominated_indices)  # Nondominated points considering a higher level
        self.centroid = self.calculate_centroid(dataset)

    def merge_clusters(self, other_cluster, dataset):
        self.subclusters.update(other_cluster.subclusters)
        self.members.update(other_cluster.members)
        self.higher_nondominants.update(other_cluster.higher_nondominants)
        self.centroid = self.calculate_centroid(dataset)

    def calculate_centroid(self, dataset):
        return np.mean([dataset[index] for index in self.members], axis=0)
    
    
def first_level_clustering(dataset, nondominated_indices):
    clusters = [NondominatedCluster(dataset, [i], {i} if i in nondominated_indices else set())
                        for i in range(len(dataset))]

    while not all(cluster.nondominants for cluster in clusters):
        if len(clusters) == 1:
            break

        centroid_matrix = np.array([cluster.centroid for cluster in clusters])
        distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

        # Track the indices of clusters already merged in this iteration
        merged_indices = set()
        absorbed_indices = set()
        for i, cluster in enumerate(clusters):
            if len(cluster.nondominants) == 0 and i not in merged_indices:
                closest_idx = np.argmin(distances[i])
                if closest_idx not in merged_indices:  # Ensure the closest cluster wasn't merged already
                    cluster.merge_clusters(clusters[closest_idx], dataset)
                    # Mark both clusters as merged
                    merged_indices.update([i, closest_idx])
                    absorbed_indices.add(closest_idx)

        # Remove merged clusters by filtering out those marked as merged
        clusters = [c for idx, c in enumerate(clusters) if idx not in absorbed_indices]

    # DEBUG: Validate that all points are accounted for
    total_indices = set.union(*(cluster.members for cluster in clusters))
    assert len(total_indices) == len(dataset)

    return clusters


def secondary_level_clustering(dataset, primary_clusters, higher_nondominated_indices):
    # Initialize secondary clusters from the primary clusters
    secondary_clusters = [AggregatedCluster(dataset, [cluster], {idx for idx in cluster.members if idx in higher_nondominated_indices})
                          for cluster in primary_clusters]

    while not all(cluster.higher_nondominants for cluster in secondary_clusters):
        if len(secondary_clusters) == 1:
            break

        centroid_matrix = np.array([cluster.centroid for cluster in secondary_clusters])
        distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

        merged_indices = set()
        absorbed_indices = set()

        for i, cluster in enumerate(secondary_clusters):
            if len(cluster.higher_nondominants) == 0 and i not in merged_indices:
                closest_idx = np.argmin(distances[i])
                if closest_idx not in merged_indices:  # Ensure the closest cluster hasn't been merged already
                    cluster.merge_clusters(secondary_clusters[closest_idx], dataset)
                    # Mark both clusters as merged
                    merged_indices.update([i, closest_idx])
                    absorbed_indices.add(closest_idx)

        # Remove absorbed clusters by filtering out those marked as absorbed
        secondary_clusters = [c for idx, c in enumerate(secondary_clusters) if idx not in absorbed_indices]

    #DEBUG: Validate that all points are accounted for
    total_indices = set.union(*(cluster.members for cluster in secondary_clusters))
    assert len(total_indices) == len(dataset)

    return secondary_clusters


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


def create_cluster_tree(secondary_clusters, evaluated_points):
    # Creating secondary level nodes
    secondary_nodes = []
    for sec_cluster in secondary_clusters:
        secondary_node = ClusterTreeNode(
            members=sec_cluster.members,
            centroid=sec_cluster.centroid,
            nondominants_3=set(),  # This will be populated from primary clusters
            nondominants_2=sec_cluster.higher_nondominants,
            children=[]
        )
                
        # Add primary clusters as children of the secondary cluster
        for primary_cluster in sec_cluster.subclusters:
            primary_node = ClusterTreeNode(
                members=primary_cluster.members,
                centroid=primary_cluster.centroid,
                nondominants_3=primary_cluster.nondominants,
                nondominants_2=set(),  # As these are primary clusters, level 2 nondominants don't apply
                children=None  # Primary clusters are leaf nodes
            )
            secondary_node.children.append(primary_node)
            secondary_node.nondominants_3.update(primary_cluster.nondominants)  # Accumulate level 1 nondominants

        secondary_nodes.append(secondary_node)

    return Tree(secondary_nodes, evaluated_points)
        
        
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
        mean, _ = self.model.predict(x)
        
        _, pareto_index = find_pareto_front(mean, return_index=True)
        # if len(pare_index) == 1:
        #     return x[pare_index]
        # else:
        #     clusters = hierarchical_clustering(Mean, pare_index)
        clusters = first_level_clustering(mean, pareto_index)

        _, pareto_second_index = find_pareto_front(mean[:, 1:], return_index=True)
        higher_clusters = secondary_level_clustering(mean, clusters, pareto_second_index)

        evaluated_points = normalize(self.model._Y.T)
        root = create_cluster_tree(higher_clusters, evaluated_points)

        candidate = root.get_candidate().nondominants_3.pop()
        return candidate



