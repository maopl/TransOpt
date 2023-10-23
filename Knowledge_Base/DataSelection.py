import GPy
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple,Union

from GPy.inference.latent_function_inference import expectation_propagation
from GPy.inference.latent_function_inference import ExactGaussianInference
from GPy.likelihoods.multioutput_likelihood import MixedNoise

from sklearn.cluster import KMeans
from sklearn.preprocessing import power_transform

from Util.Data import vectors_to_ndarray
from Util.Normalization import normalize
from Util.Kernel import construct_multi_objective_kernel
from Knowledge_Base.DataHandlerBase import selector_register
from Util.Data import vectors_to_ndarray, output_to_ndarray

def MTRankS(self,
           X:np.ndarray,
           Y:np.ndarray,
           params:Union[Dict, None] = None) -> Tuple[int, List]:
    if self.kb.get_dataset_num() < 1:
        return 0, []

    Xdim = X.shape[1]

    if isinstance(params, Dict) and 'anchor_num' in params:
        anchor_num = params['anchor_num']
    else:
        anchor_num = 0

    apcc_lst = []
    p_lst = []
    output_dim = 2
    inference_method = ExactGaussianInference()
    likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_obj_%s" % j) for y, j in
                        zip(Y, range(output_dim))]
    likelihood = MixedNoise(likelihoods_list=likelihoods_list)

    kernel =  construct_multi_objective_kernel(Xdim, output_dim=output_dim, base_kernel='RBF', rank=output_dim)

    for dataset_id in self.kb.get_all_dataset_id():
        if self.kb.get_input_vectors_by_id(dataset_id) == 0:
            continue
        dataset_info = self.kb.get_dataset_info_by_id(dataset_id)
        dataset_input_dim = dataset_info['input_dim']
        if dataset_input_dim != Xdim:
            continue
        var_name = dataset_info['variable_name']
        X_list = [vectors_to_ndarray(var_name, self.kb.get_input_vectors_by_id(dataset_id))]
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


    apcc_lst = np.array(apcc_lst)
    p_lst = np.array(p_lst)

    if np.max(np.abs(apcc_lst)) > threshold:
        return False, False,np.argmax(apcc_lst)
    else:
        if np.min(p_lst) < 0.05:
            return True, True, np.argmax(np.abs(apcc_lst))
        else:
            return True, False, np.argmax(np.abs(apcc_lst))

@selector_register('recent')
def SelectRecent(self, args, **kwards) -> Dict:
    AUX_DATA = {}
    history_task_num = self.db.get_dataset_num()
    if history_task_num < args.source_num:
        source_num = history_task_num
    else:
        source_num = args.source_num

    # 将ID转换为整数
    int_ids = [int(id) for id in self.db.get_all_dataset_id() if int(id) < int(self.dataset_id)]

    # 按降序排列ID
    sorted_ids = sorted(int_ids, reverse=True)

    # 选择最大的n个ID
    largest_ids = sorted_ids[:source_num]
    AUX_DATA['Y'] = []
    AUX_DATA['X'] = []

    for dataset_id in largest_ids:
        if dataset_id == int(self.dataset_id):
            continue

        var_name = self.db.get_var_name_by_id(str(dataset_id))
        X = vectors_to_ndarray(var_name, self.db.get_input_vectors_by_id(str(dataset_id)))
        Y = output_to_ndarray(self.db.get_output_values_by_id(str(dataset_id)))
        AUX_DATA['Y'].append(Y)
        AUX_DATA['X'].append(X)

    return {'History':AUX_DATA}