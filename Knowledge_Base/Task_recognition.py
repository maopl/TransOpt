import numpy as np
import GPy
import matplotlib.pyplot as plt
from GPy.mappings.constant import Constant
from Knowledge_Base.CKB import KnowledgeBase
from External.WGPOT.wgpot import *
from Visualization.Visual_model import Visual_task_recognition
from GPy.likelihoods.multioutput_likelihood import MixedNoise
from Model.MPGP import MPGP
from GPy import kern
from GPy import util
from paramz import ObsAr
from GPy.inference.latent_function_inference import expectation_propagation
from GPy.inference.latent_function_inference import ExactGaussianInference
from scipy.stats import spearmanr

from Knowledge_Base.Knowledge_selection import Knowledge_SimS,Knowledge_NSimS
from Util.Normalization import Norm_rank_pt,Norm_pt,Normalize_mean_std

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



def reco_WGP(X, Y, cur_name, KB:KnowledgeBase, seed, Init_num, quantile,threshold=2., Plot=True):
    if KB.len < 1:
        return True, KB.len

    if cur_name.split('_')[0] == 'Gym':
        return True, -1

    Xdim = X.shape[1]
    anchor_num = int(quantile * Init_num)
    try:
        Test_X = np.loadtxt(f'./Bench/Lifelone_env/randIni/ini_{Xdim}d_{Init_num}p_{seed}.txt')[:anchor_num]
        if len(Test_X.shape) == 1:
            Test_X = Test_X[:,np.newaxis]
    except:
        print('No init txt, to calculate WD!')
        raise

    ###construct Single GP for target data
    y_mean = np.mean(Y[:anchor_num])
    y_std = np.std(Y[:anchor_num])
    mean_std = y_mean/y_std
    Y = Y/y_std

    mf = Constant(Xdim, 1, mean_std)

    target_model = GPy.models.GPRegression(X, Y, GPy.kern.RBF(Xdim, ARD=False), mean_function=mf)
    # target_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
    # target_model.optimize_restarts(messages=False, num_restarts=1, verbose=False)
    model_list = []
    WD_list = []
    X_list = []
    Y_list = []
    name_list = []

    if Plot and Xdim==1:
        # plt_target_model = GPy.models.GPRegression(X, Y, GPy.kern.RBF(Xdim, ARD=False))
        # plt_target_model.optimize_restarts(messages=True, num_restarts=1, verbose=True)
        model_list.append(target_model)
        X_list.append(X)
        Y_list.append(Y)

    for i in range(KB.len):
        source_model = KB.model[i]
        name_list.append(KB.name[i])

        # source_model.optimize_restarts(messages=False, num_restarts=1, verbose=False)
        source_Y, source_cov_Y = source_model.predict(Test_X)
        source_gp= (source_Y, source_model.kern.K(Test_X))

        target_model['rbf.*lengthscale'] = source_model['rbf.*lengthscale'][0]
        target_model['rbf.*variance'] = source_model['rbf.*variance'][0]
        target_model['Gaussian_noise.*variance'] = source_model['Gaussian_noise.*variance'][0]

        predict_target_Y = target_model.predict(Test_X)
        predict_target_Y = Y[:anchor_num]
        target_gp = (predict_target_Y, target_model.kern.K(Test_X))

        wd_gp = Wasserstein_GP(target_gp, source_gp)

        WD_list.append(wd_gp)

        if Plot and Xdim==1:
            model_list.append(source_model)
            Y_list.append(source_Y)
            X_list.append(X)
        print(f'Source task_name:{KB.name[i]}, WD: {wd_gp}')

    WD = np.array(WD_list)
    min_wd = np.min(WD)
    min_wd_id = np.argmin(WD)
    if min_wd <= threshold:
        new_Flag = False
        task_id = min_wd_id
    else:
        new_Flag = True
        task_id = -1


    if Plot and Xdim==1:
        Visual_task_recognition(f'{cur_name}', model_list[0], X_list[0], Y_list[0], model_list[1:], X_list[1:], Y_list[1:], WD_list, name_list, Exper_floder='../LFL_experiments/test1')

    return new_Flag, task_id

def reco_MOGP(X, Y, cur_name, KB:KnowledgeBase, seed, Init_num, quantile, threshold=0.7, Plot=False):
    if KB.len < 1:
        return True, False, KB.len

    if cur_name.split('_')[0] == 'Gym':
        return True, False, -1

    Xdim = X.shape[1]
    anchor_num = int(quantile * Init_num)
    try:
        Test_X = np.loadtxt(f'./Bench/Lifelone_env/randIni/ini_{Xdim}d_{Init_num}p_{seed}.txt')[:anchor_num]
        if len(Test_X.shape) == 1:
            Test_X = Test_X[:,np.newaxis]
    except:
        print('No init txt, to calculate WD!')


    apcc_lst = []
    p_lst = []
    output_dim = 2
    # Set inference Method
    inference_method = ExactGaussianInference()
    ## Set likelihood
    likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_obj_%s" % j) for y, j in
                        zip(Y, range(output_dim))]
    likelihood = MixedNoise(likelihoods_list=likelihoods_list)

    kernel = construct_MOkernel(Xdim, output_dim=output_dim, base_kernel='RBF', rank=output_dim)

    if Plot and Xdim==1:
        f, ax = plt.subplots(KB.len+1, 1, figsize=(18, 8))

    for i in range(KB.len):
        selected_id = Knowledge_SimS(X, Y, KB,i,1)[0]
        X_list = [KB.x[i][selected_id]]
        X_list.append(X)
        if anchor_num <= 4*Xdim:
            y_mean = np.mean(Y)
            y_std = np.std(Y)
        else:
            y_mean = np.mean(Y[:anchor_num])
            y_std = np.std(Y[:anchor_num])
        Y_list = [KB.y[i][selected_id], Y]
        try:
            Y_list = Norm_pt(Y_list, y_mean, y_std)
        except:
            Y_list = [Normalize_mean_std(KB.y[i][selected_id], y_mean, y_std), Normalize_mean_std(Y, y_mean, y_std)]

        mf = None
        train_X, train_Y, output_index = util.multioutput.build_XY(X_list, Y_list)

        if i == 0:
            model = MPGP(train_X, train_Y, kernel, likelihood, Y_metadata={'output_index': output_index},
                         inference_method=inference_method, name=f'OBJ MPGP', mean_function=None)
            model['mixed_noise.Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-1)

            model.optimize_restarts(messages=False, num_restarts=5, verbose=False)
        else:
            model.update_model(False)
            if train_Y is not None:
                model.Y = ObsAr(train_Y)
                model.Y_normalized = model.Y
            if train_X is not None:
                model.X = ObsAr(train_X)
            model.mean_function=mf
            model.Y_metadata = {'output_index': output_index,
                                         'trials': np.ones(output_index.shape)}
            if isinstance(model.inference_method, expectation_propagation.EP):
                model.inference_method.reset()
            model.update_model(True)
            model.optimize_restarts(messages=False, num_restarts=1,
                                                        verbose=False)

        query_train_x = X[anchor_num:]

        noise_dict = {'output_index': np.array([0] * query_train_x.shape[0])[:, np.newaxis].astype(int)}
        predict_X = np.hstack((query_train_x, noise_dict['output_index']))
        m, v = model.predict(predict_X, Y_metadata=noise_dict, full_cov=False, include_likelihood=True)

        concat_Y = np.concatenate((Y_list[0][:anchor_num], m), axis=0)
        spcc, p =spearmanr(Y_list[1], concat_Y, axis=0)
        apcc_lst.append(spcc)
        p_lst.append(p)

        print(f'Source task_name:{KB.name[i]}, spcc: {spcc}, P_VALUE:{p}')
        if Plot and Xdim==1:
            legend = []
            legend_text= []
            test_x = np.arange(-1, 1.05, 0.005, dtype=np.float)

            noise_dict = {'output_index': np.array([0] * test_x.shape[0])[:, np.newaxis].astype(int)}
            m, v = model.predict(np.hstack((test_x[:,np.newaxis], noise_dict['output_index'])), Y_metadata=noise_dict, full_cov=False, include_likelihood=True)
            l1 = ax[i].plot(test_x, m[:, 0], 'r', linewidth=1, alpha=1)
            legend.append(l1)
            legend_text.append(f'source model')

            noise_dict = {'output_index': np.array([1] * test_x.shape[0])[:, np.newaxis].astype(int)}
            m, v = model.predict(np.hstack((test_x[:,np.newaxis], noise_dict['output_index'])), Y_metadata=noise_dict, full_cov=False, include_likelihood=True)
            l2 = ax[i].plot(test_x, m[:, 0], 'blue', linewidth=1, alpha=1)
            legend.append(l2)
            legend_text.append(f'target model')

            p1 = ax[i].plot(X[:, 0], Y[:, 0]/y_std, marker='*', color='black', linewidth=0)
            legend.append(p1)
            legend_text.append(f'target observed point')

            p2 = ax[i].plot(KB.x[i][selected_id][:, 0], KB.y[selected_id][i][:, 0], marker='*', color='green', linewidth=0)
            legend.append(p2)
            legend_text.append(f'source observed point')

            ax[i].legend(handles=legend, labels=legend_text)

    apcc_lst = np.array(apcc_lst)
    p_lst = np.array(p_lst)

    if np.max(np.abs(apcc_lst)) > threshold:
        return False, False,np.argmax(apcc_lst)
    else:
        if np.min(p_lst) < 0.05:
            return True, True, np.argmax(np.abs(apcc_lst))
        else:
            return True, False, np.argmax(np.abs(apcc_lst))


def reco_Tree(X, Y, cur_name, KB:KnowledgeBase, seed):
    if KB.len < 1:
        return True, KB.len

    fun_name = cur_name.split('_')[0]
    flag = True
    knowledge_id = -1
    for i in range(KB.len):
        if fun_name == KB.name[i].split('_')[0]:
            flag = False
            knowledge_id = i
            break

    return flag, knowledge_id