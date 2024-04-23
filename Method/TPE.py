import numpy as np
import os

from transopt.utils.Initialization import InitData
from optimizer import TPEOptimizer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def TPE(
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

    cs = CS.ConfigurationSpace()
    task_design_space = []
    for var in range(Xdim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f'x{var}', lower=-1, upper=1))

    EnvironmentChange = True

    # Set model
    while (Env.get_unsolved_num() != 0):
        if EnvironmentChange:
            EnvironmentChange = False
            opt = TPEOptimizer(config_space=cs, resultfile='a', n_init=Init)

        if Env.get_query_num() < opt._n_init:
            if Env.get_task_type() == 'Tabular':
                suggested_sample, ids = InitData(f'Tabular_{Init_method}', KB, 1, Xdim, Dty, Env=Env)
                eval_config = {f'x{d}':[suggested_sample[0][d]] for d in range(Xdim)}
                Y_ob = Env.f(suggested_sample, ids)
                results = {'loss': Y_ob}
            else:
                eval_config = opt.initial_sample()
                suggested_sample = np.array([[v for _, v in eval_config.items()]])
                Y_ob = Env.f(suggested_sample)
                results = {'loss': Y_ob}
        else:
            if Env.get_task_type() == 'Tabular':
                eval_config, idx = opt.sample_from_tabular(Env.get_all_unobserved_X(), Env.get_all_unobserved_idxs())
                suggested_sample = np.array([[v for _, v in eval_config.items()]])
                Y_ob = Env.f(suggested_sample, [idx])
                results = {'loss': Y_ob}
            else:
                eval_config = opt.sample()
                suggested_sample = np.array([[v for _, v in eval_config.items()]])
                Y_ob = Env.f(suggested_sample)
                results = {'loss': Y_ob}

        opt.update(eval_config=eval_config, results=results, runtime=0.0)




        # --- Evaluate *f* in X, augment Y and update cost function (if needed)

        # --- Augment X
        if Env.get_query_num() == 1:
            train_x = suggested_sample
            train_y = Y_ob[:,np.newaxis]
        else:
            train_x = np.vstack((train_x, suggested_sample))
            train_y = np.vstack((train_y, Y_ob))


        np.savetxt('{}/data/{}_{}/{}d/{}/{}_x.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_x)
        np.savetxt('{}/data/{}_{}/{}d/{}/{}_y.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, Env.get_current_task_name()), train_y)

        if Env.get_query_num() == Env.get_curbudget():
            EnvironmentChange=True

            Env.roll()
