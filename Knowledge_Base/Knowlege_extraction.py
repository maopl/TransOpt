import numpy as np
import GPy

from GPy.mappings.constant import Constant

from External.WGPOT.wgpot import GP_W_barycenter
from Knowledge_Base.CKB import KnowledgeBase
from Util.Normalization import Normalize, Normalize_std
from External.WGPOT.wgpot import *

from sklearn.cluster import KMeans
from GPy.inference.latent_function_inference import expectation_propagation
from GPy.inference.latent_function_inference import ExactGaussianInference
from GPy.likelihoods.multioutput_likelihood import MixedNoise
from GPy import util
from Model.MPGP import MPGP
import GPyOpt
from Acquisition.LCB import AcquisitionLCB
from Acquisition.sequential import Sequential
from Method.MultiTask import MultiTaskOptimizer

def construct_MOkernel(input_dim,output_dim, base_kernel = 'RBF', Q = 1, rank=2):
    if base_kernel == 'RBF':
        k = GPy.kern.RBF(input_dim=input_dim)
    else:
        k = GPy.kern.RBF(input_dim=input_dim)

    kernel_list = [k] * Q
    j = 1
    kk = kernel_list[0]
    K = kk.prod(
        GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], rank=rank, W=None, kappa=None,
                               name='B'), name='%s%s' % ('ICM', 0))
    for kernel in kernel_list[1:]:
        K += kernel.prod(
            GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], rank=rank, W=None,
                                   kappa=None, name='B'), name='%s%s' % ('ICM', j))
        j += 1
    return K

def Extr_barycenter(train_x, train_y, Init_point_num, knowledge_id, KB:KnowledgeBase, Seed, quantile):
    Xdim = train_x.shape[1]
    gp_list = []
    anchor_num = int(quantile * Init_point_num)

    anchor_point = train_x[:anchor_num]
    query_point = train_x[anchor_num:]

    y_mean = np.mean(train_y)
    y_std = np.std(train_y)
    mean_std = y_mean/y_std
    Y_norm = train_y/y_std

    mf = Constant(Xdim, 1, mean_std)

    target_model = GPy.models.GPRegression(train_x, Y_norm, GPy.kern.RBF(Xdim, ARD=False), mean_function=mf)
    target_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
    target_model.optimize_restarts(messages=False, num_restarts=1, verbose=False)

    source_coreset_X = KB.x[knowledge_id]
    query_coreset_point_X_id = []
    for id, x in enumerate(source_coreset_X):
        fg = False
        for ap in anchor_point:
            if (x == ap).all():
                fg = True
                break
        if fg == False:
            query_coreset_point_X_id.append(id)
    query_coreset_point_X = source_coreset_X[query_coreset_point_X_id]


    X = np.concatenate((train_x, source_coreset_X), axis=0)
    X = np.unique(X, axis=0)

    KB_model = KB.model[knowledge_id]

    Y_target_pre, _ = target_model.predict(X)
    Y_source_pre, _ = KB_model.predict(X)

    target_K_pre = target_model.kern.K(X)
    source_K_pre = KB_model.kern.K(X)

    gp_list.append((Y_source_pre, source_K_pre))
    gp_list.append((Y_target_pre, target_K_pre))
    mu, k = GP_W_barycenter(gp_list)

    source_model = GPy.models.GPRegression(X, mu, GPy.kern.RBF(Xdim, ARD=False))
    source_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
    source_model.optimize_restarts(messages=False, num_restarts=1, verbose=False)

    query_set = np.concatenate((query_point, query_coreset_point_X), axis=0)
    query_set = np.unique(query_set, axis=0)
    query_point_num = len(query_set)

    delete_num = X.shape[0] - source_coreset_X.shape[0]

    new_mu = mu.copy()
    new_k = k.copy()
    new_X = X.copy()

    for i in range(delete_num):
        max_wd = 0
        for j in range(query_point_num):
            test_x = query_set[j]
            test_id = None
            for idx, x in enumerate(new_X):
                if (test_x == x).all():
                    test_id = idx
                    break
            test_Y = np.delete(new_mu, test_id,axis=0)
            test_X = np.delete(new_X, test_id,axis=0)
            test_k = np.delete(new_k, test_id, axis=0)
            test_k = np.delete(test_k, test_id, axis=1)
            target_gp = (test_Y, test_k)
            source_K = source_model.kern.K(test_X)
            source_gp = (test_Y, source_K)

            wd = Wasserstein_GP(target_gp, source_gp)
            if max_wd < wd:
                max_wd = wd
                del_point = test_x
                del_id = test_id

        for j in range(query_point_num):
            if (query_set[j] == del_point).all():
                query_set = np.delete(query_set, j,axis=0)
                query_point_num -= 1
                break

        new_X = np.delete(new_X, del_id, axis=0)
        new_mu = np.delete(new_mu, del_id, axis=0)
        new_k = np.delete(new_k, del_id, axis=0)
        new_k = np.delete(new_k, del_id, axis=1)

    new_model = GPy.models.GPRegression(X, mu, GPy.kern.RBF(Xdim, ARD=False))
    new_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
    new_model.optimize_restarts(messages=False, num_restarts=1, verbose=False)

    return new_X, new_mu, new_model


def Extr_Kmeans(train_x, train_y, Init_point_num, knowledge_id, KB:KnowledgeBase, Seed, quantile):
    Xdim = train_x.shape[1]
    coreset_X = np.zeros(shape=train_x.shape)

    anchor_num = int(quantile * Init_point_num)
    anchor_point = train_x[:anchor_num]
    coreset_X[:anchor_num] = train_x[:anchor_num]
    query_target_point = train_x[anchor_num:]

    y_mean = np.mean(train_y)
    y_std = np.std(train_y)
    mean_std = y_mean/y_std
    Y_norm = train_y/y_std

    source_coreset_X = KB.x[knowledge_id]
    query_coreset_point_X_id = []
    for id, x in enumerate(source_coreset_X):
        fg = False
        for ap in anchor_point:
            if (x == ap).all():
                fg = True
                break
        if fg == False:
            query_coreset_point_X_id.append(id)
    query_coreset_point = source_coreset_X[query_coreset_point_X_id]

    query_set_X = np.concatenate((query_coreset_point, query_target_point),axis=0)
    # query_set_Y =
    X = np.concatenate((train_x, source_coreset_X), axis=0)
    X = np.unique(X, axis=0)

    n_cluster = train_x.shape[0]
    kmeans = KMeans(n_cluster,random_state=0).fit(X)
    labels = kmeans.labels_

    (uniq, freq) = (np.unique(labels, return_counts=True))
    counts = np.column_stack((uniq, freq))
    sorted_label = sorted(counts, key=lambda x: x[1])
    n_coreset_x = 0
    add_list = []
    while n_coreset_x <= train_x.shape[0] - anchor_num:
        temp_lst1 = []
        temp_lst2 = []
        for query_id, query_x in enumerate(query_set_X):
            for x_id, x in enumerate(X):
                if (x == query_x).all():
                    x_index = x_id
                    query_index = query_id
                    x_label = labels[x_id]
                    break
            for c in counts:
                if x_label == c[0] and c[1] == 1:
                    temp_lst1.append(x_index)
                    temp_lst2.append(query_index)
                    break

        if len(temp_lst1) + n_coreset_x <= train_x.shape[0] - anchor_num:
            add_list.extend(temp_lst1)
            query_set_X = np.delete(query_set_X, temp_lst2, axis=0)
        else:
            tmp_Y = [Y_norm[id] for id in temp_lst1]

        n_coreset_x += len(add_list)


def Extr_mean(train_x, train_y, Init_point_num, knowledge_id, KB: KnowledgeBase, Seed, quantile):
    Xdim = train_x.shape[1]
    coreset_X = np.zeros(shape=train_x.shape)

    anchor_num = int(quantile * Init_point_num)
    anchor_train_x = train_x[:anchor_num]
    query_train_x = train_x[anchor_num:]
    anchor_train_y = train_y[:anchor_num]
    query_train_y = train_y[anchor_num:]

    y_mean = np.mean(anchor_train_y)
    y_std = np.std(anchor_train_y)
    mean_std = y_mean / y_std
    anchor_train_y = anchor_train_y / y_std
    query_train_y = query_train_y / y_std

    source_coreset_X = KB.x[knowledge_id]
    source_coreset_Y = KB.y[knowledge_id]

    anchor_source_Y = source_coreset_Y[:anchor_num]

    query_source_X = source_coreset_X[anchor_num:]
    query_source_Y = source_coreset_Y[anchor_num:]


    coreset_X = anchor_train_x
    coreset_Y = 0.2*anchor_train_y + 0.8*anchor_source_Y

    query_set_X = np.concatenate((query_train_x, query_source_X), axis=0)
    query_set_Y =np.concatenate((query_train_y, query_source_Y), axis=0)
    a = np.argsort(query_set_Y,axis=0)
    maxdis_lst = []
    for x in query_set_X:
        maxdis_lst.append(np.max(np.linalg.norm(x - query_set_X)))
    maxdis_lst_sort =  np.argsort(-np.array(maxdis_lst))
    fit = []
    for i in range(query_set_X.shape[0]):
        rank_y = np.argwhere(a == i)[0][0]
        rank_d = np.argwhere(maxdis_lst_sort==i)[0][0]
        if rank_y == 0 or rank_y == (query_set_X.shape[0] - 1):
            fit.append(-1)
            continue
        if rank_d == 0:
            fit.append(-1)
            continue
        fit.append(rank_y+rank_d)
    fit_sort = np.argsort(np.array(fit))
    rest_num = train_x.shape[0] - anchor_num
    coreset_X = np.vstack((coreset_X, query_set_X[fit_sort[:rest_num]]))
    coreset_Y = np.vstack((coreset_Y, query_set_Y[fit_sort[:rest_num]]))

    mf = Constant(Xdim, 1, (np.mean(query_source_Y) + mean_std) /2)
    new_model = GPy.models.GPRegression(coreset_X, coreset_Y, GPy.kern.RBF(Xdim, ARD=False), mean_function=mf)
    new_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
    new_model.optimize_restarts(messages=False, num_restarts=5, verbose=False)

    return coreset_X,coreset_Y, new_model


def Extr_rank(X, Y, Init_point_num, knowledge_id, KB: KnowledgeBase, Seed, quantile):
    Xdim = X.shape[1]
    Target_data = {}
    Source_data = {}

    Source_data['X'] = [X]
    Source_data['Y'] = [Normalize(Y)]
    Target_data['X'] = KB.x[knowledge_id]
    Target_data['Y'] = Normalize(KB.y[knowledge_id])

    bounds =  np.array([[-1.0] * Xdim, [1.0] * Xdim])
    task_design_space = []
    for var in range(Xdim):
        var_dic = {'name': f'var_{var}', 'type': 'continuous',
                   'domain': tuple([bounds[0][var], bounds[1][var]])}
        task_design_space.append(var_dic.copy())
    Opt = MultiTaskOptimizer(Xdim, bounds=bounds, kernel='RBF', likelihood=None, acf_name=None)
    Opt.create_model('MOGP', Source_data, Target_data)
    acf_space = GPyOpt.Design_space(space=task_design_space)
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(acf_space)
    acquisition = AcquisitionLCB(Opt, acf_space, acquisition_optimizer,exploration_weight=0)

    evaluator = Sequential(acquisition)

    suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
    suggested_sample = acf_space.zip_inputs(suggested_sample)

    m, v = Opt.predict(suggested_sample)

    source_coreset_X = np.concatenate((KB.x[knowledge_id], suggested_sample),axis=0)
    source_coreset_Y = np.concatenate((KB.y[knowledge_id], np.array([[(np.min(Target_data['Y']) - 0.05) * np.std(KB.y[knowledge_id]) + np.mean(KB.y[knowledge_id])]])),axis=0)

    return source_coreset_X, source_coreset_Y, None









