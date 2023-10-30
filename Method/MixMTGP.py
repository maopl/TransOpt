import time
import numpy as np
import pandas
import gc
import GPyOpt as GPyOpt
import pickle
import os

import GPy
from  GPy.mappings.constant import Constant
from Util.Normalization import Norm_rank_pt,Norm_pt,Normalize_mean_std
from Util.Initialization import InitData
from Util import Prior
from Optimizer.MixOptimizer import MixOptimizer

from Acquisition.ConstructACF import construct_acf

from KnowledgeBase.Task_recognition import reco_MOGP,reco_Tree
from KnowledgeBase.Knowlege_extraction import Extr_barycenter,Extr_Kmeans, Extr_mean, Extr_rank
from KnowledgeBase.Knowledge_selection import  Source_selec, Source_selec_rank, Source_selec_ori

import Visualization.Visual_landscape as visual
from Gym import Gym_Metric

from Gym import Gym_func
from Util.Cpu_state import get_cpu_state

def Mix(
        Dty=np.float64,
        Plt=False,
        Init=None,
        Xdim=None,
        Env=None,
        Acf='LCB',
        Normalize_method = 'all',
        Seed=None,
        Method=None,
        model_name = 'GP',
        KB=None,
        Init_method='uniform',
        Save_mode=1,
        Exper_folder=None,
        knowledge_num = 3,
        ini_quantile = 0.9,
):

    if not os.path.exists('{}/data/{}_{}/{}d/{}/'.format(Exper_folder, Method, model_name, Xdim, Seed)):
        try:
            os.makedirs('{}/data/{}_{}/{}d/{}/'.format(Exper_folder, Method, model_name, Xdim, Seed))
        except:
            pass
    if not os.path.exists('{}/model/{}d/{}_{}/'.format(Exper_folder, Xdim, Method, model_name)):
        try:
            os.makedirs('{}/model/{}d/{}_{}/'.format(Exper_folder, Xdim, Method, model_name))
        except:
            pass
    bounds =  np.array([[-1.0] * Xdim, [1.0] * Xdim])
    task_design_space = []
    for var in range(Xdim):
        var_dic = {'name': f'var_{var}', 'type': 'continuous',
                   'domain': tuple([bounds[0][var], bounds[1][var]])}
        task_design_space.append(var_dic.copy())

    EnvironmentChange = True

    Opt = MixOptimizer(Xdim, bounds=bounds, kernel='RBF', likelihood=None, acf_name=Acf)
    objective = GPyOpt.core.task.SingleObjective(Env.f)

    Target_data = {}
    auxiliary_DATA = {}
    reco_num = {'p_right':0, 'p_false':0, 'n_right':0, 'n_false':0}

    acf_space = GPyOpt.Design_space(space=task_design_space)
    evaluator, acquisition = construct_acf(Opt, acf_space, Acfun=Acf)

    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            ## Sample current environment
            if Env.get_current_task_name().split('_')[0] == 'Gym':
                train_x = InitData('random', KB, Init, Xdim, Dty)
            else:
                train_x = InitData(Init_method, KB, Init, Xdim, Dty, seed=Seed, quantile = ini_quantile)
            train_y = Env.f(train_x)[:, np.newaxis]

            Y_mean = np.mean(train_y)
            Y_std = np.std(train_y)

            if Env.get_current_task_name().split('_')[0] == 'Gym':
                gym_match_id = Env.get_match_id()

            EnvironmentChange = False
            print(f'Target task name:{Env.get_current_task_name()}')

        Target_data['X'] = train_x
        Target_data['Y'] = Norm_pt(train_y, Y_mean, Y_std)

        if Env.get_current_task_name().split('_')[0] != 'Gym':
            if Env.get_query_num() == Init or \
                    int((Env.get_curbudget() - Init) * 0.4) == Env.get_query_num() - Init or \
                    int((Env.get_curbudget() - Init) * 0.8) == Env.get_query_num() - Init:

                # new_Flag, has_similar, knowledge_id = reco_MOGP(train_x, train_y, Env.get_current_task_name(),KB, Seed, Init, ini_quantile, threshold=0.65)
                new_Flag, knowledge_id = reco_Tree(train_x, train_y, Env.get_current_task_name(),KB, Seed)
                has_similar = False
                if new_Flag and has_similar is False:
                    print(f'Current task is a New Task!')
                    auxiliary_DATA['Y'] = []
                    auxiliary_DATA['X'] = []

                    parameters = {'lengthscale': [0.1, 2], 'variance': [0.1, 2]}
                    ls_prior = Prior.LogGaussian(np.mean(parameters['lengthscale']),np.var(parameters['lengthscale']), 'lengthscale')
                    var_prior = Prior.LogGaussian(np.mean(parameters['variance']), np.var(parameters['variance']), 'variance')
                else:
                    print(
                        f'The most similar knowledge name is {KB.name[knowledge_id]}')
                    auxiliary_DATA = Source_selec(Target_data['X'], train_y, KB, knowledge_id, knowledge_num)
                    parameters = KB.prior[knowledge_id]
                    ls_prior = Prior.LogGaussian(np.mean(parameters['lengthscale']),np.var(parameters['lengthscale']), 'lengthscale')
                    var_prior = Prior.LogGaussian(np.mean(parameters['variance']), np.var(parameters['variance']), 'variance')

                mf = None
                Opt.create_model(model_name, auxiliary_DATA, Target_data, mf, prior=[ls_prior, var_prior])
            else:
                Opt.updateModel(auxiliary_DATA, Target_data=Target_data)
        else:
            if Env.get_query_num() == Init:
                auxiliary_DATA['Y'] = []
                auxiliary_DATA['X'] = []
                mf = None
                parameters = KB.prior[gym_match_id]
                ls_prior = Prior.LogGaussian(np.mean(parameters['lengthscale']), np.var(parameters['lengthscale']),
                                             'lengthscale')
                var_prior = Prior.LogGaussian(np.mean(parameters['variance']), np.var(parameters['variance']),
                                              'variance')
                Opt.create_model(model_name, auxiliary_DATA, Target_data, mf, prior=[ls_prior, var_prior])
            else:
                Opt.updateModel(auxiliary_DATA, Target_data=Target_data)

        suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
        suggested_sample = acf_space.zip_inputs(suggested_sample)

        # --- Evaluate *f* in X, augment Y and update cost function (if needed)
        Y_new, _ = objective.evaluate(suggested_sample)

        # --- Update model parameter prior
        if True:
            ls, var = Opt.get_model_para()
            if Y_new < np.min(train_y):
                if ls > 1e-3 and ls < 10:
                    parameters['lengthscale'].append(ls)
                if var > 1e-3 and var < 10:
                    parameters['variance'].append(var)

            Opt.update_prior(parameters)

        # --- Augment X
        train_x = np.vstack((train_x, suggested_sample))
        train_y = np.vstack((train_y, Y_new))

        if Plt and Env.get_current_task_name().split('_')[0] != 'Gym':
            if Xdim == 2:
                visual.visual_contour('{}_lflT'.format(Env.get_query_num()), Env.get_current_task_name(), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample, source_data=auxiliary_DATA,method='TMTGP',
                                  dtype=Dty, Exper_folder=Exper_folder)
            elif Xdim == 1:
                visual.plot_one_dimension('{}_lfl'.format(Env.get_query_num()), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample,
                                  dtype=Dty, Exper_floder=Exper_folder)

        np.savetxt('{}/data/{}_{}/{}d/{}/{}_x.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_x)
        np.savetxt('{}/data/{}_{}/{}d/{}/{}_y.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_y)

        if Env.get_query_num() == Env.get_curbudget():
            func_name = Env.get_current_task_name()

            traj_perf = Gym_Metric.traj_metric(train_y, Init)
            trend_perf = Gym_Metric.trend_metric(train_x, train_y)
            rug_perf = Gym_Metric.rug_metric(train_x, train_y)
            Metric ={'rug':rug_perf, 'trend':trend_perf, 'traj':traj_perf}

            coreset_X = train_x
            coreset_Y = train_y

            if Env.get_current_task_name().split('_')[0] != 'Gym' and KB.len != 0:
                if new_Flag == False and Env.get_current_task_name().split('_')[0] == KB.name[knowledge_id].split('_')[
                    0]:
                    reco_num['p_right'] += 1
                elif new_Flag == True:
                    ff = False
                    for name in KB.name:
                        if Env.get_current_task_name().split('_')[0] == name.split('_')[0]:
                            ff = True
                            continue
                    if ff:
                        reco_num['p_false'] += 1
                    else:
                        reco_num['n_right'] += 1
                elif new_Flag == False and Env.get_current_task_name().split('_')[0] != \
                        KB.name[knowledge_id].split('_')[0]:
                    reco_num['n_false'] += 1

                df = pandas.DataFrame(reco_num, index=[0])
                df.to_excel(f'./count/reco_num_{Xdim}d_seed{Seed}.xlsx')

            if func_name.split('_')[0]  == 'Gym':
                KB.update_prior(gym_match_id, parameters)
            else:
                if new_Flag:
                    knowledge_name = func_name.split('_')[0]+f'_{Env.get_current_task_id()}'
                    KB.add(knowledge_name, 'LFL', coreset_X, coreset_Y, Opt.obj_model, parameters, Metric)
                else:
                    KB.update(knowledge_id, coreset_X, coreset_Y, None, parameters, Metric)

            # Gym_num = 0
            # if func_name.split('_')[0] != 'Gym' and get_cpu_state() == 'Free':
            #     for i in range(2):
            #         Gym_setting = Gym_func.construct_setting(KB, knowledge_id)
            #         gym_name = f'Gym_{Gym_num}'
            #         gym_func = Gym_func.Gym(gym_name, knowledge_id, Gym_setting, input_dim=Xdim, Seed=Seed)
            #         Env.add_task_to_next(gym_func, 22*Xdim)
            #         Gym_num += 1

            EnvironmentChange=True

            Env.roll()
