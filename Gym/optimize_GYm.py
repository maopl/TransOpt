import numpy as np
import GPyOpt as GPyOpt
import pickle
import os


from transopt.Utils.Normalization  import Normalize
from transopt.Utils.Initialization import InitData
from transopt.Optimizer import RestartOptimizer
from transopt.Optimizer import construct_acf
from transopt.Utils import Prior


def GymOpt(
        Dty=np.float64,
        Plt=False,
        Init=None,
        Max_eval = None,
        Xdim=None,
        prob=None,
        Acf='EI',
        Seed=None,
        Method=None,
        model_name = 'GP',
        GKB=None,
        Save_mode=1,
        Exper_folder=None,
):

    os.makedirs('{}/data/{}_{}/{}d/{}/'.format(Exper_folder, Method, model_name, Xdim, Seed), exist_ok=True)
    os.makedirs('{}/model/{}_{}/{}d/'.format(Exper_folder, Method, model_name, Xdim), exist_ok=True)

    bounds =  np.array([[-1.0] * Xdim, [1.0] * Xdim])
    task_design_space = []
    for var in range(Xdim):
        var_dic = {'name': f'var_{var}', 'type': 'continuous',
                   'domain': tuple([bounds[0][var], bounds[1][var]])}
        task_design_space.append(var_dic.copy())

    Opt = RestartOptimizer(Xdim, bounds=bounds, kernel='RBF', likelihood=None, acf_name=Acf)
    objective = GPyOpt.core.task.SingleObjective(prob.f)

    Target_data = {}

    # Set model
    Init_method = 'uniform'
    train_x, _ = InitData(f'Continuous_{Init_method}', GKB, Init, Xdim, Dty)
    train_y = prob.f(train_x)
    train_y = train_y[:, np.newaxis]

    for iter in range(Init, Max_eval):
        train_y_norm = Normalize(train_y)
        Target_data['X'] = train_x
        Target_data['Y'] = train_y_norm

        if iter == Init:
            prior = {'lengthscale': Prior.Gamma(0.5, 1, 'lengthscale'), 'variance': Prior.Gamma(0.5, 1, 'variance')}
            Opt.create_model('GP', Target_data, [prior['lengthscale'], prior['variance']])
            acf_space = GPyOpt.Design_space(space=task_design_space)
            evaluator, acquisition = construct_acf(Opt, acf_space, Acfun=Acf)
        else:
            Opt.updateModel(Target_data)

            suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
            suggested_sample = acf_space.zip_inputs(suggested_sample)
            Y_new, _ = objective.evaluate(suggested_sample)


            ls, var = Opt.get_model_para()
            if Y_new < np.min(train_y):
                if ls > 1e-3 and ls < 10:
                    Opt.update_prior(ls, 'lengthscale')
                if var > 1e-3 and var < 10:
                    Opt.update_prior(var, 'variance')

            # --- Augment X
            train_x = np.vstack((train_x, suggested_sample))
            train_y = np.vstack((train_y, Y_new))


            np.savetxt('{}/data/{}_{}/{}d/{}/{}_x.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, prob.name), train_x)
            np.savetxt('{}/data/{}_{}/{}d/{}/{}_y.txt'.format(Exper_folder, Method, model_name, Xdim, Seed, prob.name), train_y)
            prior = {'lengthscale': Opt.get_model_prior('lengthscale'), 'variance': Opt.get_model_prior('variance')}
            GKB.add(prob.name, 'GYM', train_x, train_y, Opt.obj_model, prior, match_id=prob.match_id)
            if Save_mode == 1:
                with open('{}/model/{}_{}/{}d/{}_GKB.txt'.format(Exper_folder, Method, model_name,Xdim, Seed), 'wb') as f:  # 打开文件
                    pickle.dump(GKB, f)
    return GKB