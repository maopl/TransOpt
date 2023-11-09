import time
import numpy as np
import GPyOpt as GPyOpt
import pickle
import os
import pandas as pds

from Util.Normalization import Norm_rank_pt,Norm_pt,Normalize_mean_std, Normalize
from Util.Initialization import InitData
from Optimizer.VanillaBOa import VBOOptimizer
from Optimizer.Acquisition.ConstructACF import construct_acf

import Visualization.Visual_landscape as visual
from torch.quasirandom import SobolEngine
from External.hebo.design_space.design_space import DesignSpace


def quasi_sample(sobol, n, space, fix_input=None):
    samp = sobol.draw(n)
    samp = samp * (space.opt_ub - space.opt_lb) + space.opt_lb
    x = samp[:, :space.num_numeric]
    xe = samp[:, space.num_numeric:]
    for i, n in enumerate(space.numeric_names):
        if space.paras[n].is_discrete_after_transform:
            x[:, i] = x[:, i].round()
    df_samp = space.inverse_transform(x, xe)
    if fix_input is not None:
        for k, v in fix_input.items():
            df_samp[k] = v
    return df_samp
import torch
from torch import Tensor, FloatTensor, LongTensor
from External.hebo.models.scalers import TorchMinMaxScaler, TorchStandardScaler

def fit_scaler(space, Xc: FloatTensor, y: FloatTensor):
    xscaler = TorchMinMaxScaler((-1, 1))
    yscaler = TorchStandardScaler()
    if Xc is not None and Xc.shape[1] > 0:
        if space is not None:
            cont_lb = space.opt_lb[:space.num_numeric].view(1, -1).float()
            cont_ub = space.opt_ub[:space.num_numeric].view(1, -1).float()
            xscaler.fit(torch.cat([Xc, cont_lb, cont_ub], dim=0))
        else:
            xscaler.fit(Xc)
    yscaler.fit(y)

    return xscaler, yscaler


def trans(xscaler, yscaler, Xc : Tensor, Xe : Tensor, y : Tensor = None):
    if Xc is not None and Xc.shape[1] > 0:
        Xc_t = xscaler.transform(Xc)
    else:
        Xc_t = torch.zeros(Xe.shape[0], 0)

    if Xe is None or Xe.shape[1] == 0:
        Xe_t = torch.zeros(Xc.shape[0], 0)
    else:
        Xe_t = one_hot(Xe.long())
    Xall = torch.cat([Xc_t, Xe_t], dim = 1)

    if y is not None:
        y_t = yscaler.transform(y)
        return Xall.numpy(), y_t.numpy()
    return Xall.numpy()

def test_BO(
        Dty=np.float64,
        Plt=False,
        Init=None,
        Xdim=None,
        Env=None,
        Acf='EI',
        Normalize_method = 'all',
        Seed=None,
        Method=None,
        model_name = 'GP',
        KB=None,
        Init_method='random',
        Save_mode=1,
        Exper_folder=None,
        Terminate_criterion = None,
        source_task_num = 2,
):

    if not os.path.exists('{}/data/{}_{}/{}d/{}/'.format(Exper_folder, Method, model_name, Xdim, Seed)):
        os.makedirs('{}/data/{}_{}/{}d/{}/'.format(Exper_folder, Method, model_name, Xdim, Seed))
    if not os.path.exists('{}/model/{}d/{}_{}/'.format(Exper_folder, Xdim, Method, model_name)):
        os.makedirs('{}/model/{}d/{}_{}/'.format(Exper_folder, Xdim, Method, model_name))

    bounds =  np.array([[-1.0] * Xdim, [1.0] * Xdim])
    space = []
    for var in range(Xdim):
        var_dic = {'name': f'var_{var}', 'type': 'num',
                   'lb': bounds[0][var], 'ub': bounds[1][var]}
        space.append(var_dic.copy())

    space = DesignSpace().parse(space)

    task_design_space = []
    for var in range(Xdim):
        var_dic = {'name': f'var_{var}', 'type': 'continuous',
                   'domain': tuple([bounds[0][var], bounds[1][var]])}
        task_design_space.append(var_dic.copy())


    EnvironmentChange = True


    soboel_engine = SobolEngine(space.num_paras, scramble = True, seed = Seed)
    Opt = VBOOptimizer(Xdim, bounds=bounds, kernel='RBF', likelihood=None, acf_name=Acf)
    objective = GPyOpt.core.task.SingleObjective(Env.f)

    Target_data = {}

    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            ## Sample current environment
            train_x = quasi_sample(soboel_engine, 1, space)
            for i in range(1, Init):
                train_x = np.concatenate((train_x, quasi_sample(soboel_engine, 1, space)))
            train_y = Env.f(train_x)
            train_y = train_y[:, np.newaxis]

            # train_y_norm = Normalize(train_y)
            train_y_norm = Norm_pt(train_y, np.mean(train_y), np.std(train_y))
            Target_data['X'] = train_x
            Target_data['Y'] = train_y_norm
            X = pds.DataFrame(train_x,columns=space.para_names)

            X, Xe = space.transform(X)
            train_y_norm = torch.FloatTensor(train_y_norm)
            xscaler, y_scaler = fit_scaler(space, X, train_y_norm)
            X, train_y_norm = trans(xscaler,y_scaler,X, Xe, train_y_norm)
            Target_data['X'] = X
            Target_data['Y'] = train_y_norm

            Opt.create_model(model_name, Target_data)

            acf_space = GPyOpt.Design_space(space=task_design_space)
            evaluator, acquisition = construct_acf(Opt, acf_space, Acfun=Acf)

            EnvironmentChange = False
        else:
            # train_y_norm = Normalize(train_y)
            train_y_norm = Norm_pt(train_y, np.mean(train_y), np.std(train_y))
            Target_data['X'] = train_x
            Target_data['Y'] = train_y_norm

            X = pds.DataFrame(train_x, columns=space.para_names)
            X, Xe = space.transform(X)
            train_y_norm = torch.FloatTensor(train_y_norm)
            xscaler, y_scaler = fit_scaler(space, X, train_y_norm)
            X, train_y_norm = trans(xscaler,y_scaler,X, Xe, train_y_norm)
            Target_data['X'] = X
            Target_data['Y'] = train_y_norm

            Opt.create_model(model_name, Target_data)

            acf_space = GPyOpt.Design_space(space=task_design_space)
            evaluator, acquisition = construct_acf(Opt, acf_space, Acfun=Acf)

        suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)

        suggested_sample = acf_space.zip_inputs(suggested_sample)

        # --- Evaluate *f* in X, augment Y and update cost function (if needed)
        Y_new, _ = objective.evaluate(suggested_sample)



        # --- Augment X
        train_x = np.vstack((train_x, suggested_sample))
        train_y = np.vstack((train_y, Y_new))

        if Plt:
            if Xdim == 2:
                visual.visual_contour('{}_lfl'.format(Env.get_query_num()), Env.get_current_task_name(), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample, method='GP', conformal=False,
                                  dtype=Dty, Exper_folder=Exper_folder)
            elif Xdim == 1:
                visual.plot_one_dimension('{}_lfl'.format(Env.get_query_num()), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample,conformal=False,
                                  dtype=Dty, Exper_floder=Exper_folder)

        np.savetxt('{}/data/{}_{}/{}d/{}/{}_x.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_x)
        np.savetxt('{}/data/{}_{}/{}d/{}/{}_y.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_y)
        KB.add(Env.get_current_task_name(), 'LFL', train_x, train_y)
        if Save_mode == 1:
            with open('{}/model/{}d/{}_{}/{}_KB.txt'.format(Exper_folder, Xdim, Method, model_name, Seed), 'wb') as f:  # 打开文件
                pickle.dump(KB, f)

        if Env.get_query_num() == Env.get_curbudget():
            # visual.visual_selection('{}_lflT'.format(Env.get_query_num()), Env.get_current_task_name(), Env, search_history, loss_history,
            #                         train_x, train_y, Init, Method='BO',
            #                         dtype=Dty, Exper_folder=Exper_folder)
            EnvironmentChange=True
            Env.roll()

