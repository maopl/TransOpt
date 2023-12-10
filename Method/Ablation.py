import numpy as np
import pandas
import GPyOpt as GPyOpt
import os

from transopt.Utils.Normalization import Norm_pt
from transopt.Utils.Initialization import InitData
from Util import Prior
from Optimizer import MOGPOptimizer
from Optimizer import MultiTaskOptimizer

from Optimizer import construct_acf

from transopt.KnowledgeBase.Task_recognition import reco_MOGP
from transopt.KnowledgeBase.Knowledge_selection import  Source_selec

import Visualization.Visual_landscape as visual
from Gym import Gym_Metric
from Gym import Generator
import pickle
from transopt.Utils.Normalization  import Normalize
from collections import defaultdict

def Aba1(
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
        GKB = None,
        Init_method='uniform',
        Save_mode=1,
        Exper_folder=None,
        knowledge_num = 3,
        ini_quantile = 0.9,
):


    os.makedirs('{}/data/{}_{}/{}d/{}/'.format(Exper_folder, Method, model_name, Xdim, Seed), exist_ok=True)
    os.makedirs('{}/model/{}d/{}_{}/'.format(Exper_folder, Xdim, Method, model_name), exist_ok=True)

    bounds =  np.array([[-1.0] * Xdim, [1.0] * Xdim])
    task_design_space = []
    for var in range(Xdim):
        var_dic = {'name': f'var_{var}', 'type': 'continuous',
                   'domain': tuple([bounds[0][var], bounds[1][var]])}
        task_design_space.append(var_dic.copy())

    EnvironmentChange = True
    Opt = MOGPOptimizer(Xdim, bounds=bounds, kernel='Matern', likelihood=None, acf_name=Acf)
    objective = GPyOpt.core.task.SingleObjective(Env.f)

    Target_data = {}
    History_DATA = {}
    reco_num = {'p_right':0, 'p_false':0, 'n_right':0, 'n_false':0}

    acf_space = GPyOpt.Design_space(space=task_design_space)
    evaluator, acquisition = construct_acf(Opt, acf_space, Acfun='LCB_LFL')

    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            ## Sample current environment
            if Env.get_task_type() == 'Tabular':
                train_x, ids = InitData(f'Tabular_random', KB, Init, Xdim, Dty, Env=Env)
                train_y = Env.f(train_x, ids)
            else:
                train_x, _ = InitData(f'Continuous_{Init_method}', KB, Init, Xdim, Dty, seed=Seed, quantile=ini_quantile)
                train_y = Env.f(train_x)

            train_y = train_y[:,np.newaxis]

            Y_mean = np.mean(train_y)
            Y_std = np.std(train_y)

            EnvironmentChange = False
            print(f'Target task name:{Env.get_current_task_name()}')

        Target_data['X'] = train_x
        Target_data['Y'] = Norm_pt(train_y, Y_mean, Y_std)

        if Env.get_query_num() == Init or \
                int((Env.get_curbudget() - Init) * 0.4) == Env.get_query_num() - Init or \
                int((Env.get_curbudget() - Init) * 0.8) == Env.get_query_num() - Init:

            if Env.get_task_type() == 'Tabular':
                new_Flag, has_similar, knowledge_id = reco_MOGP(train_x, train_y, Env.get_current_task_name(),KB, Seed, Init, 0, threshold=0.65)
            else:
                new_Flag, has_similar, knowledge_id = reco_MOGP(train_x, train_y, Env.get_current_task_name(),KB, Seed, Init, ini_quantile, threshold=0.65)


            GYM_data = {'X':[], 'Y':[]}

            if new_Flag and has_similar is False:
                print(f'Current task is a New Task!')
                History_DATA['Y'] = []
                History_DATA['X'] = []
                prior = {'lengthscale':Prior.Gamma(0.5, 1, 'lengthscale'), 'variance':Prior.Gamma(0.5, 1, 'variance')}
            else:
                print(
                    f'The most similar knowledge name is {KB.name[knowledge_id]}')
                ##Select auxiliary data for current task
                History_DATA = Source_selec(Target_data['X'], train_y, KB, knowledge_id, knowledge_num)
                prior = KB.prior[knowledge_id]

            mf = None
            Opt.create_model(model_name, History_DATA, GYM_data, Target_data, mf, prior=[prior['lengthscale'], prior['variance']])
        else:
            # Auxillary_data = combine_auxillary_data(History_DATA, GYM_data)
            Opt.updateModel(History_DATA, GYM_data, Target_data=Target_data)

        if Env.get_task_type() == 'Tabular':
            suggested_index, acq_value = evaluator.recommend_tabular(Env.get_all_unobserved_X(), Env.get_all_unobserved_idxs())
            suggested_sample = Env.get_var([suggested_index])
            Y_new = Env.f(suggested_sample, [suggested_index])
        elif Env.get_task_type() == 'Continuous':
            suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
            suggested_sample = acf_space.zip_inputs(suggested_sample)
            Y_new, _ = objective.evaluate(suggested_sample)

        # --- Update model parameter prior
        ls, var = Opt.get_model_para()
        if Y_new < np.min(train_y):
            if ls > 1e-3 and ls < 10:
                Opt.update_prior(ls, 'lengthscale')
            if var > 1e-3 and var < 10:
                Opt.update_prior(var, 'variance')

        # --- Augment X
        train_x = np.vstack((train_x, suggested_sample))
        train_y = np.vstack((train_y, Y_new))

        if Plt:
            if Xdim == 2:
                visual.visual_contour('{}_lflT'.format(Env.get_query_num()), Env.get_current_task_name(), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample, source_data=Auxillary_data,method='TMTGP',
                                  dtype=Dty, Exper_folder=Exper_folder)
            elif Xdim == 1:
                visual.plot_one_dimension('{}_lfl'.format(Env.get_query_num()), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample,
                                  dtype=Dty, Exper_floder=Exper_folder)

        np.savetxt('{}/data/{}_{}/{}d/{}/{}_x.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_x)
        np.savetxt('{}/data/{}_{}/{}d/{}/{}_y.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_y)
        if Save_mode == 1:
            with open('{}/model/{}d/{}_{}/{}_KB.txt'.format(Exper_folder, Xdim, Method, model_name, Seed), 'wb') as f:  # 打开文件
                pickle.dump(KB, f)

        traj_perf = Gym_Metric.traj_metric(train_y, Init)

        if Env.get_query_num() == Env.get_curbudget():
            func_name = Env.get_current_task_name()

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

            prior = {'lengthscale':Opt.get_model_prior('lengthscale'), 'variance':Opt.get_model_prior('variance')}

            if new_Flag:
                knowledge_name = func_name.split('_')[0]+f'_{Env.get_current_task_id()}'
                KB.add(knowledge_name, 'LFL', coreset_X, coreset_Y, Opt.obj_model, prior, None)
            else:
                KB.update(knowledge_id, coreset_X, coreset_Y, None, prior, None)

            if Save_mode == 1:
                with open('{}/model/{}d/{}_{}/{}_KB.txt'.format(Exper_folder, Xdim, Method, model_name, Seed),
                          'wb') as f:  # 打开文件
                    pickle.dump(KB, f)

            EnvironmentChange=True

            Env.roll()


def Aba2(
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
        ini_quantile = 0.9,
):

    if not os.path.exists('{}/data/{}_{}/{}d/{}/'.format(Exper_folder, Method, model_name, Xdim, Seed)):
        os.makedirs('{}/data/{}_{}/{}d/{}/'.format(Exper_folder, Method, model_name, Xdim, Seed))
    if not os.path.exists('{}/model/{}d/{}_{}/'.format(Exper_folder, Xdim, Method, model_name)):
        os.makedirs('{}/model/{}d/{}_{}/'.format(Exper_folder, Xdim, Method, model_name))

    bounds =  np.array([[-1.0] * Xdim, [1.0] * Xdim])
    task_design_space = []
    for var in range(Xdim):
        var_dic = {'name': f'var_{var}', 'type': 'continuous',
                   'domain': tuple([bounds[0][var], bounds[1][var]])}
        task_design_space.append(var_dic.copy())

    EnvironmentChange = True

    Opt = MultiTaskOptimizer(Xdim, bounds=bounds, kernel='RBF', likelihood=None, acf_name=Acf)
    objective = GPyOpt.core.task.SingleObjective(Env.f)

    Target_data = {}
    Meta_DATA = defaultdict(list)


    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            search_history = {'Y':[], 'Y_cov':[]}
            loss_history = {'Train':[], 'Test':[]}
            ## Sample current environment
            if Env.get_task_type() == 'Tabular':
                train_x, ids = InitData(f'Tabular_random', KB, Init, Xdim, Dty, Env=Env)
                train_y = Env.f(train_x, ids)
            else:
                train_x, _ = InitData(f'Continuous_{Init_method}', KB, Init, Xdim, Dty, seed=Seed, quantile=ini_quantile)
                train_y = Env.f(train_x)
            train_y = train_y[:, np.newaxis]

            train_y_norm = Normalize(train_y)
            Target_data['X'] = train_x
            Target_data['Y'] = train_y_norm

            if int((Env.get_curbudget() - Init) * 0.4) >= Env.get_query_num() - Init:
                G = Generator.PseudoPointsGenerator(n_components=int(Xdim))
                X_generated = G.generate_from_fft(Target_data['X'], Target_data['Y'][:,0],
                                                         target_samples=len(Target_data['X']) * 2)
                pseudo_Y = G.Set_Y_by_gaussian(Target_data['X'], train_y[:, 0], test_X= X_generated)
                GYM_data = {'X': [X_generated], 'Y': [pseudo_Y]}
            else:
                GYM_data = {'X':[], 'Y':[]}

            if KB.len == 0 and len(GYM_data['X'])==0:
                Opt.create_model('GP', Meta_DATA, Target_data)
            elif KB.len > 0 and len(GYM_data['X'])==0:
                ##Select source data for current environment
                if KB.len < source_task_num:
                    source_num = KB.len
                else:
                    source_num = source_task_num
                Meta_DATA['Y']=[Normalize(KB.y[-1-i][0]) for  i in range(source_num)]
                Meta_DATA['X']=[KB.x[-1-i][0] for i in range(source_num)]

                Opt.create_model(model_name, Meta_DATA, Target_data)
            elif KB.len == 0 and len(GYM_data['X'])>0:

                Opt.create_model(model_name, GYM_data, Target_data)
            else:
                ##Select source data for current environment
                if KB.len < source_task_num:
                    source_num = KB.len
                else:
                    source_num = source_task_num
                Meta_DATA['Y']=[Normalize(KB.y[-1-i][0]) for  i in range(source_num)]
                Meta_DATA['X']=[KB.x[-1-i][0] for i in range(source_num)]
                Meta_DATA['Y'].append(GYM_data['Y'][0])
                Meta_DATA['X'].append(GYM_data['X'][0])
                Opt.create_model(model_name, Meta_DATA, Target_data)

            acf_space = GPyOpt.Design_space(space=task_design_space)
            evaluator, acquisition = construct_acf(Opt, acf_space, Acfun=Acf)

            EnvironmentChange = False
        else:
            train_y_norm = Normalize(train_y)
            Target_data['X'] = train_x
            Target_data['Y'] = train_y_norm

            Opt.updateModel(Meta_DATA, Target_data)

        if Env.get_task_type() == 'Tabular':
            suggested_index, acq_value = evaluator.recommend_tabular(Env.get_all_unobserved_X(), Env.get_all_unobserved_idxs())
            suggested_sample = Env.get_var([suggested_index])
            Y_new = Env.f(suggested_sample, [suggested_index])
        elif Env.get_task_type() == 'Continuous':
            suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
            suggested_sample = acf_space.zip_inputs(suggested_sample)
            Y_new, _ = objective.evaluate(suggested_sample)

        # pred, pred_cov = Opt.predict(suggested_sample)
        # search_history['Y'].append(pred[0][0])
        # search_history['Y_cov'].append(pred_cov[0][0])

        # pred_all, cov_all = Opt.predict(train_x)
        # loss_history['Train'].append(np.sqrt(np.mean((pred_all - Target_data['Y']) ** 2)))
        #
        # test_x = 2 * sobol_seq.i4_sobol_generate(Xdim, 10000) - 1
        # pred_test_all, cov_test_all = Opt.predict(test_x)
        # Env.query_num_lock()
        # test_y = Env.f(test_x)[:, np.newaxis]
        # Env.query_num_unlock()
        # tmean = np.mean(train_y)
        # tstd = np.std(train_y)
        # test_y = (test_y - tmean) / tstd
        # loss_history['Test'].append(np.sqrt(np.mean((pred_test_all - test_y) ** 2)))

        # --- Augment X
        train_x = np.vstack((train_x, suggested_sample))
        train_y = np.vstack((train_y, Y_new))

        if Plt:
            if Xdim == 2:
                visual.visual_contour('{}_lfl'.format(Env.get_query_num()), Env.get_current_task_name(), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample, method='GP',
                                  dtype=Dty, Exper_folder=Exper_folder)
            elif Xdim == 1:
                visual.plot_one_dimension('{}_lfl'.format(Env.get_query_num()), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample,
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



def toy(
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

    Opt = MOGPOptimizer(Xdim, bounds=bounds, kernel='RBF', likelihood=None, acf_name=Acf)
    objective = GPyOpt.core.task.SingleObjective(Env.f)

    Target_data = {}
    auxiliary_DATA = {}

    acf_space = GPyOpt.Design_space(space=task_design_space)
    evaluator, acquisition = construct_acf(Opt, acf_space, Acfun='LCB_LFL')

    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            ## Sample current environment

            train_x = InitData(Init_method, KB, Init, Xdim, Dty, seed=Seed, quantile = ini_quantile)
            train_y = Env.f(train_x)[:, np.newaxis]

            EnvironmentChange = False
            print(f'Target task name:{Env.get_current_task_name()}')

        Target_data['X'] = train_x
        Target_data['Y'] = Normalize(train_y)

        if Env.get_query_num() == Init or \
                int((Env.get_curbudget() - Init) * 0.4) == Env.get_query_num() - Init or \
                int((Env.get_curbudget() - Init) * 0.8) == Env.get_query_num() - Init:

            # new_Flag, has_similar, knowledge_id = reco_MOGP(train_x, train_y, Env.get_current_task_name(),KB, Seed, Init, ini_quantile, threshold=0.65)
            new_Flag, knowledge_id = reco_Tree(train_x, train_y, Env.get_current_task_name(),KB, Seed)

            if new_Flag:
                print(f'Current task is a New Task!')
                auxiliary_DATA['Y'] = []
                auxiliary_DATA['X'] = []

            else:
                print(
                    f'The most similar knowledge name is {KB.name[knowledge_id]}')
                temp_Y= KB.y[knowledge_id][-knowledge_num:]
                auxiliary_DATA['Y']=[Normalize(y) for y in temp_Y]
                auxiliary_DATA['X']=KB.x[knowledge_id][-knowledge_num:]

            mf = None
            prior = {'lengthscale': Prior.Gamma(0.5, 1, 'lengthscale'), 'variance': Prior.Gamma(0.5, 1, 'variance')}
            Opt.create_model(model_name, auxiliary_DATA, Target_data, mf, prior=[prior['lengthscale'], prior['variance']])
        else:
            Opt.updateModel(auxiliary_DATA, Target_data=Target_data)

        suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
        suggested_sample = acf_space.zip_inputs(suggested_sample)

        # --- Evaluate *f* in X, augment Y and update cost function (if needed)
        Y_new, _ = objective.evaluate(suggested_sample)

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
            coreset_X = train_x
            coreset_Y = train_y


            if new_Flag:
                knowledge_name = func_name.split('_')[0]+f'_{Env.get_current_task_id()}'
                KB.add(knowledge_name, 'LFL', coreset_X, coreset_Y, Opt.obj_model, None, None)
            else:
                KB.update(knowledge_id, coreset_X, coreset_Y, None, None, None)

            EnvironmentChange=True

            Env.roll()
