import copy

import numpy as np
import GPyOpt as GPyOpt
import os

from transopt.Utils.Normalization  import Normalize
from transopt.Utils.Initialization import InitData
from transopt_external.transfergpbo.models import TaskData

from Optimizer import construct_acf

import Visualization.Visual_landscape as visual
from Gym import Gym_Metric

from Optimizer import MTTree

from transopt.KnowledgeBase.Task_recognition import reco_Tree

def LFL_Tree(
        Dty=np.float64,
        Plt=False,
        Init=None,
        Xdim=None,
        Env=None,
        Acf='LCB',
        Normalize_method = 'all',
        Seed=None,
        Method=None,
        model_name = 'Tree',
        KB=None,
        Init_method='uniform',
        Save_mode=1,
        Exper_folder=None,
        knowledge_num = 3,
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

    model = MTTree()

    objective = GPyOpt.core.task.SingleObjective(Env.f)

    Target_data = {}
    auxiliary_DATA = {}
    reco_num = {'p_right':0, 'p_false':0, 'n_right':0, 'n_false':0}

    acf_space = GPyOpt.Design_space(space=task_design_space)
    evaluator, acquisition = construct_acf(model, acf_space, Acfun=Acf)

    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            ## Sample current environment

            train_x = InitData(Init_method, KB, Init, Xdim, Dty, seed=Seed, quantile = ini_quantile)
            train_y = Env.f(train_x)[:, np.newaxis]

            Target_data['X'] = train_x
            Target_data['Y'] = Normalize(train_y)

            EnvironmentChange = False

        else:
            Target_data['X'] = train_x
            Target_data['Y'] = Normalize(train_y)

        new_Flag, knowledge_id = reco_Tree(train_x, train_y, Env.get_current_task_name(),KB, Seed)

        if new_Flag:
            auxiliary_DATA['X'] = []
            auxiliary_DATA['Y'] = []
        else:
            auxiliary_DATA['X'] = copy.deepcopy(KB.x[knowledge_id])
            auxiliary_DATA['Y'] = [Normalize(Y) for Y in KB.y[knowledge_id]]

        training_X = auxiliary_DATA['X']
        training_X.append(Target_data['X'])
        tmp_x = []
        for X in training_X:
            for x in X:
                tmp_x.append(x)
        training_Y = auxiliary_DATA['Y']
        training_Y.append(Target_data['Y'])
        tmp_y = []
        for Y in training_Y:
            for y in Y:
                tmp_y.append(y)

        training_data = TaskData(X=np.array(tmp_x), Y=np.array(tmp_y))
        model.fit(training_data)

        suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
        suggested_sample = acf_space.zip_inputs(suggested_sample)

        # --- Evaluate *f* in X, augment Y and update cost function (if needed)
        Y_new, _ = objective.evaluate(suggested_sample)

        # --- Augment X
        train_x = np.vstack((train_x, suggested_sample))
        train_y = np.vstack((train_y, Y_new))

        if Plt and Env.get_current_task_name().split('_')[0] != 'Gym':
            if Xdim == 2:
                visual.visual_contour('{}_lflT'.format(Env.get_query_num()), Env.get_current_task_name(), model, Env,
                                  acquisition, train_x, train_y, suggested_sample, method='Tree',
                                  dtype=Dty, Exper_folder=Exper_folder)
            elif Xdim == 1:
                visual.plot_one_dimension('{}_lflT'.format(Env.get_query_num()), model, Env,
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

            if new_Flag:
                knowledge_name = func_name.split('_')[0]
                if func_name.split('_')[0]  == 'Gym':
                    pass
                else:
                    KB.add(knowledge_name, 'LFLT', coreset_X, coreset_Y, None, None, None)
            else:
                KB.update(knowledge_id, coreset_X, coreset_Y, None,  None, None)


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
