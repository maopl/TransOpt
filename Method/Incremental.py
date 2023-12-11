import numpy as np
import GPyOpt as GPyOpt
import pickle
import os


from transopt.utils.Normalization  import Normalize
from transopt.utils.Initialization import InitData
from Optimizer import IncrementalOptimizer
from Optimizer import construct_acf

import Visualization.Visual_landscape as visual

from transopt_external.transfergpbo.models import TaskData

from collections import defaultdict
def Incremental(
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
    task_design_space = []
    for var in range(Xdim):
        var_dic = {'name': f'var_{var}', 'type': 'continuous',
                   'domain': tuple([bounds[0][var], bounds[1][var]])}
        task_design_space.append(var_dic.copy())

    EnvironmentChange = True

    Opt = IncrementalOptimizer(Xdim, bounds=bounds, kernel='RBF', likelihood=None, acf_name=Acf)
    objective = GPyOpt.core.task.SingleObjective(Env.f)

    if KB.len <= source_task_num:
        source_task_num = KB.len

    Target_data = {}
    Meta_DATA = defaultdict(list)


    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            ## Sample current environment
            if Env.get_task_type() == 'Tabular':
                train_x, ids = InitData(f'Tabular_{Init_method}', KB, Init, Xdim, Dty, Env=Env)
                train_y = Env.f(train_x, ids)
            else:
                train_x, _ = InitData(f'Continuous_{Init_method}', KB, Init, Xdim, Dty)
                train_y = Env.f(train_x)
            train_y = train_y[:, np.newaxis]

            train_y_norm = Normalize(train_y)
            Target_data['X'] = train_x
            Target_data['Y'] = train_y_norm

            if KB.len == 0:
                Opt.create_model('GP', Meta_DATA, Target_data)
            elif KB.len == 1:
                ##Select source data for current environment
                Meta_DATA['Y']=[Normalize(KB.y[-1][0])]
                Meta_DATA['X']=[KB.x[-1][0]]
                Opt.create_model(model_name,Meta_DATA, Target_data)
            else:
                meta_data = {0:TaskData(X=KB.x[-1][0], Y=Normalize(KB.y[-1][0]))}
                Opt.meta_add(meta_data)
                Opt.updateModel(Target_data)

            acf_space = GPyOpt.Design_space(space=task_design_space)
            evaluator, acquisition = construct_acf(Opt, acf_space, Acfun=Acf)

            EnvironmentChange = False
        else:
            train_y_norm = Normalize(train_y)
            Target_data['X'] = train_x
            Target_data['Y'] = train_y_norm

            Opt.updateModel(Target_data)

        if Env.get_task_type() == 'Tabular':
            suggested_index, acq_value = evaluator.recommend_tabular(Env.get_all_unobserved_X(), Env.get_all_unobserved_idxs())
            suggested_sample = Env.get_var([suggested_index])
            Y_new = Env.f(suggested_sample, [suggested_index])
        elif Env.get_task_type() == 'Continuous':
            suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
            suggested_sample = acf_space.zip_inputs(suggested_sample)
            Y_new, _ = objective.evaluate(suggested_sample)

        # --- Augment X
        train_x = np.vstack((train_x, suggested_sample))
        train_y = np.vstack((train_y, Y_new))

        if Plt:
            if Xdim == 2:
                visual.plot_contour('{}_lfl'.format(Env.get_query_num()), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample,
                                  dtype=Dty, Exper_floder=Exper_folder)
            elif Xdim == 1:
                visual.plot_one_dimension('{}_lfl'.format(Env.get_query_num()), Opt, Env,
                                  acquisition, train_x, train_y, suggested_sample,
                                  dtype=Dty, Exper_floder=Exper_folder)

        np.savetxt('{}/data/{}_{}/{}d/{}/{}_x.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_x)
        np.savetxt('{}/data/{}_{}/{}d/{}/{}_y.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_y)
        KB.add(Env.get_current_task_name(), 'INC', train_x, train_y)
        if Save_mode == 1:
            with open('{}/model/{}d/{}_{}/{}_KB.txt'.format(Exper_folder, Xdim, Method, model_name, Seed), 'wb') as f:  # 打开文件
                pickle.dump(KB, f)

        if Env.get_query_num() == Env.get_curbudget():
            EnvironmentChange=True
            Env.roll()

