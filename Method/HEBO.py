import numpy as np
import pickle
import os

from External.hebo.design_space.design_space import DesignSpace
from External.hebo.optimizers.hebo import HEBO as hebo

def HEBO(
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
        var_dic = {'name': f'var_{var}', 'type': 'num',
                   'lb': bounds[0][var], 'ub': bounds[1][var]}
        task_design_space.append(var_dic.copy())

    space = DesignSpace().parse(task_design_space)

    EnvironmentChange = True

    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            ## Sample current environment
            Opt = hebo(space, rand_sample=Init, scramble_seed=Seed)

            EnvironmentChange = False

        if Env.get_task_type() == 'Tabular':
            suggested_sample = Opt.suggest(n_suggestions=1,)
            suggested_index = Env.get_idx(suggested_sample)
            Y_ob = Env.f(suggested_sample, [suggested_index])
        else:
            suggested_sample = Opt.suggest(n_suggestions=1)
            Y_ob = Env.f(np.array(suggested_sample))


        Opt.observe(suggested_sample, Y_ob)

        # --- Augment X
        if Env.get_query_num() == 1:
            train_x = np.array(suggested_sample)
            train_y = Y_ob[:,np.newaxis]
        else:
            train_x = np.vstack((train_x, np.array(suggested_sample)))
            train_y = np.vstack((train_y, Y_ob))

        np.savetxt('{}/data/{}_{}/{}d/{}/{}_x.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_x)
        np.savetxt('{}/data/{}_{}/{}d/{}/{}_y.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_y)
        KB.add(Env.get_current_task_name(), 'LFL', train_x, train_y)
        if Save_mode == 1:
            with open('{}/model/{}d/{}_{}/{}_KB.txt'.format(Exper_folder, Xdim, Method, model_name, Seed), 'wb') as f:  # 打开文件
                pickle.dump(KB, f)

        if Env.get_query_num() == Env.get_curbudget():
            EnvironmentChange=True
            Env.roll()

