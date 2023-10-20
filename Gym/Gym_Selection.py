import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import power_transform

def selection_by_id(GKB, knowledge_id, Xdim, selection_num = 44):
    # clu_num = int(X.shape[1] * 6)
    clu_num = 2 * Xdim
    aux_Data = {}
    for func_index, match_id in enumerate(GKB.match_id):
        if func_index == knowledge_id:
            Knowledge_X = GKB.x[func_index][0].copy()
            Y_mean = np.mean(GKB.y[func_index])
            Y_std = np.std(GKB.y[func_index])

            selected_X = []
            kmn = KMeans(n_clusters=int(clu_num), random_state=0)
            kmn.fit(GKB.x[func_index][0])
            lables = kmn.labels_
            centers = kmn.cluster_centers_

            for c_id, center in enumerate(centers):
                tmp_id = []
                for x_id, x in enumerate(GKB.x[func_index][0]):
                    if lables[x_id] == c_id:
                        tmp_id.append(x_id)

                # Randomly choose a subset of data from the cluster
                num_to_select = min(1, len(tmp_id))
                selected_indices = np.random.choice(tmp_id, size=num_to_select, replace=False)
                selected_X.extend(selected_indices)

            aux_Data['X'] = [Knowledge_X[selected_X]]
            Union_Y = GKB.y[func_index][0]
            Union_Y = power_transform((Union_Y - Y_mean) / Y_std, method='yeo-johnson')
            aux_Data['Y'] = [Union_Y[selected_X]]

            return aux_Data


    return {'X':[], 'Y':[]}