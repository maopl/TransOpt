import copy

import numpy as np
from KnowledgeBase.CKB import KnowledgeBase
from sklearn.cluster import KMeans
from Util.Normalization import normalize
from sklearn.preprocessing import power_transform
def Knowledge_randS(Target_data, task_id, KB:KnowledgeBase, number):
    auxiliary_DATA = {}

    coreset_X = KB.x[task_id]
    coreset_Y = KB.y[task_id]
    Xdim = coreset_X.shape[1]
    a = np.random.randint(0, coreset_X.shape[0], size=(number))

    auxiliary_DATA['Y'] = [Normalize(coreset_Y[a])]
    auxiliary_DATA['X'] = [coreset_X[a]]
    return coreset_X[a], coreset_Y[a]

def Data_gene_all(Target_data,task_id, KB:KnowledgeBase, number):
    auxiliary_DATA = {}

    coreset_X = KB.x[task_id]
    coreset_Y = KB.y[task_id]

    return coreset_X, coreset_Y

def Data_gene_Kmeans(Target_data, task_id, KB:KnowledgeBase, number):
    auxiliary_DATA = {}
    selected_id = []

    train_x = Target_data['X']

    coreset_X = KB.x[task_id]
    coreset_Y = KB.y[task_id]
    min_y = np.min(coreset_Y, axis=0)
    Xdim = coreset_X.shape[1]
    coreset_num = coreset_X.shape[0]

    kmn = KMeans(n_clusters=int(number), random_state=0)
    kmn.fit(coreset_Y)
    lables = kmn.labels_
    centers = kmn.cluster_centers_
    for c_id, center in enumerate(centers):
        dis = []
        tmp_id = []
        for y_id, y in enumerate(coreset_Y):
            if lables[y_id] == c_id:
                if y == min_y:
                    dis.append(100000)
                    tmp_id.append(y_id)
                    continue
                dis.append(np.min(np.linalg.norm(coreset_X[y_id] - train_x, axis=1)))
                tmp_id.append(y_id)

        selected_id.append(tmp_id[np.argmax(dis)])

    return coreset_X[selected_id], coreset_Y[selected_id]




def Data_gene_rank(Target_data, task_id, KB:KnowledgeBase, number):
    auxiliary_DATA = {}
    selected_id = []

    train_x = Target_data['X']

    coreset_X = KB.x[task_id]
    coreset_Y = KB.y[task_id]
    min_y = np.min(coreset_Y, axis=0)
    Xdim = coreset_X.shape[1]
    coreset_num  = coreset_X.shape[0]

    kmn = KMeans(n_clusters=int(number), random_state=0)
    kmn.fit(coreset_Y)
    lables = kmn.labels_
    centers = kmn.cluster_centers_
    for c_id, center in enumerate(centers):
        dis = []
        tmp_id = []
        for y_id, y in enumerate(coreset_Y):
            if lables[y_id] == c_id:
                if y == min_y:
                    dis.append(100000)
                    tmp_id.append(y_id)
                    continue
                dis.append(np.min(np.linalg.norm(coreset_X[y_id] - train_x, axis=1)))
                tmp_id.append(y_id)

        selected_id.append(tmp_id[np.argmax(dis)])

    return coreset_X[selected_id], coreset_Y[selected_id]


def Knowledge_SimS(X, Y, KB:KnowledgeBase, Knowledge_id, Knowledge_num):
    Knowledge_X = KB.x[Knowledge_id]
    Knowledge_Y = KB.y[Knowledge_id]
    Knowledge_len = len(Knowledge_Y)

    mean_dis = []
    for set_id in range(Knowledge_len):
        dis = []
        for x_id, x in enumerate(Knowledge_X[set_id]):
            dis.append(np.min(np.linalg.norm(Knowledge_X[set_id][x_id] - X, axis=1)))
        mean_dis.append(np.mean(dis))

    selected_id = np.argsort(mean_dis, axis=0)[:Knowledge_num]

    return selected_id



def Knowledge_NSimS(X, Y, KB:KnowledgeBase, Knowledge_id, Knowledge_num):
    Knowledge_X = KB.x[Knowledge_id]
    Knowledge_Y = KB.y[Knowledge_id]
    Knowledge_len = len(Knowledge_Y)

    mean_dis = []
    for set_id in range(Knowledge_len):
        dis = []
        for x_id, x in enumerate(Knowledge_X[set_id]):
            dis.append(np.min(np.linalg.norm(Knowledge_X[set_id][x_id] - X, axis=1)))
        mean_dis.append(np.mean(dis))

    selected_id = np.argsort(mean_dis, axis=0)[-Knowledge_num:]
    return selected_id


def Source_selec(X, Y, KB:KnowledgeBase, Knowledge_id, Knowledge_num):
    Knowledge_X = KB.x[Knowledge_id]
    Knowledge_Y = KB.y[Knowledge_id]
    Knowledge_len = len(Knowledge_Y)

    clu_num = int(X.shape[1] * 6)

    mean_dis = []

    for set_id in range(Knowledge_len):
        dis = []
        for x_id, x in enumerate(Knowledge_X[set_id]):
            dis.append(np.min(np.linalg.norm(Knowledge_X[set_id][x_id] - X, axis=1)))
        mean_dis.append(np.mean(dis))

    selected_id = np.argsort(mean_dis, axis=0)[-Knowledge_num:]

    aux_Data ={'X':[], 'Y':[]}
    Union_X = copy.deepcopy(X)
    Union_Y = copy.deepcopy(Y)
    Y_mean = np.mean(Y[:7 * X.shape[1]])
    Y_std = np.std(Y[:7 * X.shape[1]])

    selected_id_num = [X.shape[0]]
    source_data_num = X.shape[0]

    for set_id in selected_id:
        selected_X = []
        kmn = KMeans(n_clusters=int(clu_num), random_state=0)
        kmn.fit(Knowledge_X[set_id])
        lables = kmn.labels_
        centers = kmn.cluster_centers_

        for c_id, center in enumerate(centers):
            dis = []
            tmp_id = []
            for x_id, x in enumerate(Knowledge_X[set_id]):
                if lables[x_id] == c_id:
                    dis.append(np.min(np.linalg.norm(Knowledge_X[set_id][x_id] - Union_X, axis=1)))
                    tmp_id.append(x_id)

            selected_X.append(tmp_id[np.argmax(dis)])

        Union_X = np.concatenate((Union_X, Knowledge_X[set_id][selected_X]), axis=0)
        Union_Y = np.concatenate((Union_Y, Knowledge_Y[set_id][selected_X]), axis=0)
        aux_Data['X'].append(Knowledge_X[set_id][selected_X])
        source_data_num += len(selected_X)
        selected_id_num.append(source_data_num)

    Union_Y = power_transform((Union_Y-Y_mean)/Y_std, method = 'yeo-johnson')

    for id, num in enumerate(selected_id_num[:-1]):
        aux_Data['Y'].append(Union_Y[num:selected_id_num[id+1]])

    return aux_Data

def Source_selec_rank(X, Y, KB:KnowledgeBase, Knowledge_id, Knowledge_num):
    Knowledge_X = KB.x[Knowledge_id]
    Knowledge_Y = KB.y[Knowledge_id]
    Knowledge_len = len(Knowledge_Y)

    clu_num = int(X.shape[1] * 6)

    mean_dis = []

    for set_id in range(Knowledge_len):
        dis = []
        for x_id, x in enumerate(Knowledge_X[set_id]):
            dis.append(np.min(np.linalg.norm(Knowledge_X[set_id][x_id] - X, axis=1)))
        mean_dis.append(np.mean(dis))

    selected_id = np.argsort(mean_dis, axis=0)[-Knowledge_num:]

    aux_Data ={'X':[], 'Y':[]}
    Union_X = copy.deepcopy(X)
    Union_Y = copy.deepcopy(Y)
    Y_mean = np.mean(Y[:8 * X.shape[1]])
    Y_std = np.std(Y[:8 * X.shape[1]])

    selected_id_num = [X.shape[0]]
    source_data_num = X.shape[0]

    for set_id in selected_id:
        selected_X = []
        kmn = KMeans(n_clusters=int(clu_num), random_state=0)
        kmn.fit(Knowledge_X[set_id])
        lables = kmn.labels_
        centers = kmn.cluster_centers_

        for c_id, center in enumerate(centers):
            dis = []
            tmp_id = []
            for x_id, x in enumerate(Knowledge_X[set_id]):
                if lables[x_id] == c_id:
                    dis.append(np.min(np.linalg.norm(Knowledge_X[set_id][x_id] - Union_X, axis=1)))
                    tmp_id.append(x_id)

            selected_X.append(tmp_id[np.argmax(dis)])

        Union_X = np.concatenate((Union_X, Knowledge_X[set_id][selected_X]), axis=0)

        sorted_indices = np.argsort(Knowledge_Y[set_id], axis=0)[:, 0]
        rank_array = np.zeros(shape=len(Knowledge_Y[set_id]))
        rank_array[sorted_indices] = np.arange(1, len(Knowledge_Y[set_id]) + 1)
        rank_array = rank_array[selected_X, np.newaxis]

        Union_Y = np.concatenate((Union_Y, power_transform(rank_array, method='yeo-johnson')), axis=0)
        aux_Data['X'].append(Knowledge_X[set_id][selected_X])
        source_data_num += len(selected_X)
        selected_id_num.append(source_data_num)


    for id, num in enumerate(selected_id_num[:-1]):
        aux_Data['Y'].append(Union_Y[num:selected_id_num[id+1]])

    return aux_Data



def Source_selec_ori(X, Y, KB:KnowledgeBase, Knowledge_id, Knowledge_num):
    Knowledge_X = KB.x[Knowledge_id]
    Knowledge_Y = KB.y[Knowledge_id]
    Knowledge_len = len(Knowledge_Y)

    clu_num = int(X.shape[1] * 6)

    mean_dis = []

    for set_id in range(Knowledge_len):
        dis = []
        for x_id, x in enumerate(Knowledge_X[set_id]):
            dis.append(np.min(np.linalg.norm(Knowledge_X[set_id][x_id] - X, axis=1)))
        mean_dis.append(np.mean(dis))

    selected_id = np.argsort(mean_dis, axis=0)[-Knowledge_num:]

    aux_Data ={'X':[], 'Y':[]}
    Union_X = copy.deepcopy(X)
    Union_Y = copy.deepcopy(Y)


    selected_id_num = [X.shape[0]]
    source_data_num = X.shape[0]

    for set_id in selected_id:
        selected_X = []
        kmn = KMeans(n_clusters=int(clu_num), random_state=0)
        kmn.fit(Knowledge_X[set_id])
        lables = kmn.labels_
        centers = kmn.cluster_centers_

        for c_id, center in enumerate(centers):
            dis = []
            tmp_id = []
            for x_id, x in enumerate(Knowledge_X[set_id]):
                if lables[x_id] == c_id:
                    dis.append(np.min(np.linalg.norm(Knowledge_X[set_id][x_id] - Union_X, axis=1)))
                    tmp_id.append(x_id)

            selected_X.append(tmp_id[np.argmax(dis)])

        Union_X = np.concatenate((Union_X, Knowledge_X[set_id][selected_X]), axis=0)

        Union_Y = np.concatenate((Union_Y, Normalize(Knowledge_Y[set_id])[selected_X]), axis=0)
        aux_Data['X'].append(Knowledge_X[set_id][selected_X])
        source_data_num += len(selected_X)
        selected_id_num.append(source_data_num)


    for id, num in enumerate(selected_id_num[:-1]):
        aux_Data['Y'].append(Union_Y[num:selected_id_num[id+1]])

    return aux_Data

if __name__ == '__main__':
    pass