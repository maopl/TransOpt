import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import warnings
from Bench.Synthetic import SyntheticBenchmark
from Util.Normalization import Normalize_mean_std, Normalize
from itertools import product
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE."



def visual_selection(file_name, title, problem, search_history, loss_history,
                   train_x, train_y, init,
                   method, Seed: int = 0, test_size=101, show: bool = True,
                   dtype=np.float64, Exper_folder=None):
    plt.clf()
    f, ax = plt.subplots(2, 3, figsize=(24, 12))
    # f, ax = plt.subplots(figsize=(12, 12))

    # Test points every 0.02 in [0,1]
    optimizers = problem.optimizers
    xgrid_0, xgrid_1 = np.meshgrid(np.linspace(-1, 1, test_size, dtype=dtype),
                                      np.linspace(-1, 1, test_size, dtype=dtype))

    pred_y = np.array(search_history['Y'])
    pred_cov = np.array(search_history['Y_cov'])

    train_y = Normalize(train_y)


    ax_p = ax[0][0]
    ax_p.plot(list(range(train_y[init:].shape[0])), loss_history['Train'], marker='o', linestyle='-', color='red', linewidth=1, alpha=1, label='Training MSE')
    ax_p.plot(list(range(train_y[init:].shape[0])), loss_history['Test'], marker='o', linestyle='-', color='b', linewidth=1, alpha=1, label='Test MSE')
    ax_p.set_ylabel('MSE')
    ax_p.set_xlabel('Function evaluation number')
    ax_p.set_title('Training Loss &. Test loss for each iteration')
    ax_p.legend()

    ax_p = ax[0][1]
    ax_p.plot(list(range(train_y[init:].shape[0])), train_y[init:], marker='o', linestyle='-', color='red', linewidth=1, alpha=1, label='Real value')
    ax_p.errorbar(list(range(train_y[init:].shape[0])), pred_y[:], yerr=pred_cov[:], linestyle='-', fmt='o-', color='b', label='Prediction with covariance')
    ax_p.legend()
    ax_p.set_ylabel('Y')
    ax_p.set_xlabel('Function evaluation number')
    ax_p.set_title('Reale Value VS. Prediction Value on the next evaluation point')
    ax_p.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_p.set_xlim(0, train_y[init:].shape[0])
    ax_p.set_xlabel('Function evaluation number')

    ax_p = ax[0][2]

    def best(x, y):
        best_y = np.ones((1, len(y)))
        best_x = np.ones((len(x), len(x[0])))
        best_index = []
        for j in range(len(y)):
            best_index.append(np.argmin(y[:j + 1]))
            best_y[0, j] = y[best_index[-1]]
            best_x[j] = x[best_index[-1]]
        return best_x, best_y, best_index
    best_x, best_y, _ = best(train_x, train_y)
    ax_p.plot(list(range(train_y[init:].shape[0])), best_y[0, init:], marker='o', linestyle='-', color='red', linewidth=1, alpha=1)
    ax_p.set_ylabel('Y')
    ax_p.set_xlabel('Function evaluation number')
    ax_p.set_title('Optimization Trajectory')
    ax_p.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_p.set_xlim(0, train_y[init:].shape[0])
    ax_p.set_xlabel('Function evaluation number')



    ax_p = ax[1][0]
    dis = np.linalg.norm(train_x - optimizers, axis=1)
    ax_p.plot(list(range(train_y[init:].shape[0])), dis[init:], marker='o', linestyle='-', color='b', linewidth=1, alpha=1)
    ax[1, 0].sharex(ax[0, 1])
    ax_p.set_ylabel('Distance')
    ax_p.set_xlabel('Function evaluation number')
    ax_p.set_title('X Distance to the global optima')

    ax_p = ax[1][1]
    def min_dis(x, S):
        min_ = np.min(np.linalg.norm(S - x, axis=1))
        return min_
    min_distance = [np.linalg.norm(train_x[init+ i - 1] - train_x[init+i]) for i in range(train_x[init:].shape[0])]
    min_distance[0] = min_dis(train_x[init], train_x[:init])
    ax_p.plot(list(range(train_y[init:].shape[0])), min_distance, marker='o', linestyle='-', color='b', linewidth=1, alpha=1)
    ax[1, 1].sharey(ax[1, 0])
    ax[1, 1].sharex(ax[0, 1])
    ax_p.set_title('X Distance to the last query point')
    ax_p.set_xlabel('Function evaluation number')


    ax_p = ax[1][2]
    boundaries = np.array([[1,1,1,1,1], [-1,-1,-1,-1,-1]])
    min_distances = [np.min(np.min(np.abs(x - boundaries), axis=1), axis=0) for x in train_x[init:]]
    ax_p.plot(list(range(train_y[init:].shape[0])), min_distances, marker='o', linestyle='-', color='b', linewidth=1, alpha=1)
    # ax[1, 2].sharey(ax[1, 0])
    ax[1, 2].sharex(ax[0, 1])
    ax_p.set_title('X Distance to the boundary')
    ax_p.set_xlabel('Function evaluation number')

    if not os.path.exists('{}/figs/selection/{}/{}'.format(Exper_folder, method, Seed)):
        os.makedirs('{}/figs/selection/{}/{}'.format(Exper_folder, method, Seed))

    plt.savefig('{}/figs/selection/{}/{}/{}.png'.format(Exper_folder, method, Seed, title), format='png')
    plt.close()


def plot_contour(file_name, model, Env, ac_model,
                        train_x, train_y, Ac_candi, test_size=101,
                       dtype=np.float64, Exper_floder=None):
    # Initialize plots
    f, y_ax = plt.subplots(1, 2, figsize=(16, 6))

    # Test points every 0.02 in [0,1]
    problem = Env.get_current_problem()
    bounds = problem.bounds
    # optimizers = problem.optimizers
    xgrid_0, xgrid_1 = np.meshgrid(np.linspace(bounds[0][0], bounds[1][0], test_size, dtype=dtype),
                                      np.linspace(bounds[0][1], bounds[1][1], test_size, dtype=dtype))
    test_x = np.concatenate((xgrid_0.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1)),
                        xgrid_1.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1))),
                       axis=1)

    # Make predictions - one task at a time
    # We control the task we cae about using the indices
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)

    observed_pred_y, observed_corv = model.predict(test_x)
    observed_pred_y = observed_pred_y.reshape(xgrid_0.shape)
    observed_corv = observed_corv.reshape(xgrid_0.shape)

    # Calculate the true value
    test_y = problem.f(test_x)
    test_y = test_y.reshape(xgrid_0.shape)

    mean = np.mean(train_y)
    std = np.std(train_y)
    test_y = Normalize_mean_std(test_y, mean, std)
    train_y_temp = Normalize(train_y)

    # Calculate EI for the problem

    test_ei = ac_model._compute_acq(test_x)
    GRID_Best = np.max(test_ei)
    GRID_BestScore = test_x[np.argmax(test_ei)]

    test_ei = test_ei.reshape(xgrid_0.shape)

    # Define plotting function
    def ax_plot(title, ax, train_y, train_x, test_y, test_x, test_ei, best_ei, test_size, observed_pred_y, observed_corv, Ac_x):
        # Get lower and upper confidence bounds
        # lower, upper = rand_var.confidence_region()
        # Visualization training data as black stars
        # train_mean = observed_pred_y.reshape(xgrid_0.shape)
        ax[0].plot(train_x[:, 0], train_x[:, 1], 'k*')
        # Predictive mean as blue line
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax[0].contour(
                xgrid_0,
                xgrid_1,
                observed_pred_y,
                cmap=cm.Blues
            )
            min_loc_1 = (int(np.argmin(observed_pred_y) / test_size),
                         np.remainder(np.argmin(observed_pred_y), test_size))
            ax[0].plot(xgrid_0[min_loc_1],
                       xgrid_1[min_loc_1], 'b*')
            # True value as red line
            ax[0].contour(xgrid_0, xgrid_1,
                          test_y, cmap=cm.Reds)
            min_loc_2 = (int(np.argmin(test_y) / test_size),
                         np.remainder(np.argmin(test_y), test_size))
            # ax[0].plot(optimizers[0], optimizers[1], 'r*')
            # Shade in confidence
            # ax.fill_between(test_x.squeeze().detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
            # EI in gray line
            ax[0].contour(xgrid_0, xgrid_1,
                          test_ei, cmap=cm.Greens)
            max_loc = (int(np.argmax(test_ei) / test_size),
                       np.remainder(np.argmax(test_ei), test_size))
            ax[0].plot(xgrid_0[max_loc],
                       xgrid_1[max_loc],
                       'g*')
            ax[0].plot(Ac_x[0][0], Ac_x[0][1],
                       color='orange', marker='*', linewidth=0)

        # ax.set_ylim([-3, 3])
        ax[0].legend(['Observed Data', 'Prediction', 'True f(x)', 'EI', 'Candidate'])
        ax[0].set_xlim([bounds[0][0], bounds[1][0]])
        ax[0].set_ylim([bounds[0][1], bounds[1][1]])
        num_sample = train_x.shape[0]
        ax[0].set_title(title + ' Sample(' + str(num_sample) + ')')

        # plt.subplot(grid[0, 2])
        # ax[1].text(1, 1, "Prediction:\n"
        #                  "x1={:.4}, x2={:.4}, y={:.4}\n"
        #                  "\n"
        #                  "True f(x):\n"
        #                  "x1={:.4}, x2={:.4}, y={:.4}\n"
        #                  "\n"
        #                  "EI:\n"
        #                  "x1={:.4}, x2={:.4}, y={:.4}\n"
        #                  "\n"
        #                  "Candidate:\n"
        #                  "x1={:.4}, x2={:.4}, y={:.4}".format(
        #     xgrid_0[min_loc_1],
        #     xgrid_1[min_loc_1],
        #     np.min(observed_pred_y),
        #     xgrid_0[min_loc_2],
        #     xgrid_1[min_loc_2],
        #     np.min(test_y),
        #     xgrid_0[max_loc],
        #     xgrid_1[max_loc],
        #     np.max(test_ei),
        #     Ac_x[0][0],
        #     Ac_x[0][1],
        #     Ac_y[0][0]
        # ), fontsize=12)
        ax[1].axis([0, 10, 0, 10])
        ax[1].axis('off')

    ax_plot(f'{Env.get_query_num()}_LFL', y_ax, train_y_temp, train_x, test_y, test_x, test_ei, GRID_BestScore, test_size, observed_pred_y, observed_corv,
            Ac_candi)
    plt.grid()

    if not os.path.exists('{}/figs/contour/{}/{}'.format(Exper_floder, model.name, f'{Env.get_current_task_name()}_LFL')):
        os.makedirs('{}/figs/contour/{}/{}'.format(Exper_floder, model.name, f'{Env.get_current_task_name()}_LFL'))

    plt.savefig('{}/figs/contour/{}/{}/{}.png'.format(Exper_floder, model.name, f'{Env.get_current_task_name()}_LFL', file_name), format='png')
    plt.close()



def visual_contour(file_name, title, model, problem, ac_model,
                   train_x, train_y, Ac_candi,
                   method, source_data = None, Seed: int = 0, test_size=101, show: bool = True,conformal=True,
                   dtype=np.float64, Exper_folder=None):
    # Initialize plots
    plt.clf()
    if source_data is None:
        f, ax = plt.subplots(2, 2, figsize=(12, 12))
    else:
        f, ax = plt.subplots(2, 3, figsize=(24, 12))



    # Test points every 0.02 in [0,1]
    problem.query_num_lock()
    # optimizers = problem.optimizers
    xgrid_0, xgrid_1 = np.meshgrid(np.linspace(-1, 1, test_size, dtype=dtype),
                                      np.linspace(-1, 1, test_size, dtype=dtype))
    test_x = np.concatenate((xgrid_0.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1)),
                        xgrid_1.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1))),
                       axis=1)



    # Calculate the true value
    test_y = problem.f(test_x)[:,np.newaxis]
    mean = np.mean(train_y)
    std = np.std(train_y)
    test_y = (test_y - mean) /std
    problem.query_num_unlock()

    # Make predictions - one task at a time
    # We control the task we cae about using the indices
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    pred_y, pred_corv = model.predict(test_x)
    if conformal and model.qhats is not  None:
        _, CP = model.conformal_prediction(test_x)
        CP = CP.reshape(xgrid_0.shape)
    # observed_pred_y = observed_pred_y.reshape(xgrid_0.shape)
    pred_corv = pred_corv.reshape(xgrid_0.shape)


    # Calculate acquisition function for the problem
    test_ei = ac_model._compute_acq(test_x)
    test_ei = test_ei.reshape(xgrid_0.shape)

    # Define plotting function
    def ax_plot(title, ax, train_x, plot_y, test_size, Seed, cmap):
        # Get lower and upper confidence bounds
        # lower, upper = rand_var.confidence_region()
        # Visual training data as black stars
        # train_mean = observed_pred_y.reshape(xgrid_0.shape)
        ax.plot(train_x[:, 0], train_x[:, 1], 'k*')
        # Predictive mean as blue line
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h1 = ax.contourf(
                xgrid_0,
                xgrid_1,
                plot_y,
                np.arange(-3,3.5,0.5),
                cmap = cmap,
            )
            c1 = plt.colorbar(h1, ax=ax)
            # ax.clabel(C, inline=True)
            min_loc_1 = (int(np.argmin(plot_y) / test_size),
                         np.remainder(np.argmin(plot_y), test_size))
            ax.plot(xgrid_0[min_loc_1],
                       xgrid_1[min_loc_1], 'b*')
            # Shade in confidence
            # ax.fill_between(test_x.squeeze().detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
            # EI in gray line


        ax.set_xlim([-1, 1])
        # ax.set_ylim([bounds[0][1], bounds[1][1]])
        num_sample = train_x.shape[0]
        ax.set_title(title )

        # plt.subplot(grid[0, 2])

    # PLot true contour in the left plot
    ax_plot(title+ ' at Seed=' + str(Seed) + ' Sample(' + str(train_x.shape[0]) + ')', ax[0][0], train_x, test_y.reshape(xgrid_0.shape), test_size, Seed, cm.Reds)

    ax_plot('Prediction', ax[0][1], train_x, pred_y.reshape(xgrid_0.shape), test_size, Seed, cm.Blues)

    def ax_plot_ei(title, ax, train_x, plot_ei, Ac_x, cmap):
        # Predictive mean as blue line
        h1 = ax.contourf(xgrid_0, xgrid_1,
                    plot_ei,np.arange(-3,3.5,0.5), cmap=cmap)
        c1 = plt.colorbar(h1, ax=ax)
        max_loc = (int(np.argmax(plot_ei) / test_size),
                   np.remainder(np.argmax(plot_ei), test_size))
        ax.plot(xgrid_0[max_loc],
                   xgrid_1[max_loc],
                   'g*')
        ax.plot(Ac_x[0][0], Ac_x[0][1],
                   color='orange', marker='*', linewidth=0)
        num_sample = train_x.shape[0]
        ax.set_title(title)
    ax_plot_ei('Acquisition Function', ax[1][1], train_x, test_ei, Ac_candi ,cm.Greens)

    # PLot covariance contour in the last row
    ax_plot('Prediction covariance', ax[1][0], train_x, pred_corv, test_size, Seed, cm.Blues)
    if conformal and model.qhats is not None:
        ax_plot('Conformal prediction', ax[1][2], train_x, CP, test_size, Seed, cm.Blues)

    if source_data is not None:
        ax_p = ax[0][2]
        source_X = source_data['X']
        source_Y = source_data['Y']
        source_num = len(source_Y)
        markers = ['s', 'o', '^', 'p', '*','v', '<']

        for data_id, X in enumerate(source_X):
            colors = source_Y[data_id]
            ax_p.scatter(X[:, 0], X[:, 1], c=colors, cmap='Reds', s=100, marker=markers[data_id])

    plt.grid()
    # if (show):
    #     plt.show()

    if not os.path.exists('{}/figs/contour/{}/{}/{}'.format(Exper_folder, model.name, Seed, title)):
        os.makedirs('{}/figs/contour/{}/{}/{}'.format(Exper_folder, model.name, Seed, title))

    plt.savefig('{}/figs/contour/{}/{}/{}/{}.png'.format(Exper_folder, model.name, Seed, title, file_name), format='png')
    plt.close()

def plot_one_dimension(file_name, model, Env, ac_model,
                        train_x, train_y, Ac_candi,conformal=True,
                       dtype=np.float64, Exper_floder=None):
    # Initialize plots
    f, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Test points every 0.02 in [0,1]

    problem = Env.get_current_problem()

    bounds = problem.bounds
    opt_x = problem.optimizers
    opt_val = problem.optimal_value
    test_x = np.arange(-1, 1.05, 0.005, dtype=dtype)


    # Make predictions - one task at a time
    # We control the task we cae about using the indices
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    observed_pred_y, observed_corv = model.predict(test_x[:,np.newaxis])

    if conformal and model.qhats is not None:
        _, cfv, = model.conformal_prediction(test_x[:, np.newaxis])
    # observed_corv = model.var_predict(test_x[:,np.newaxis])
    # Calculate the true value
    test_y = problem.f(test_x[:,np.newaxis])

    y_mean = np.mean(train_y)
    y_std = np.std(train_y)
    mean_std = y_mean / y_std

    test_y = (test_y - y_mean)  / y_std
    train_y_temp = (train_y  - y_mean)/ y_std


    # Calculate EI for the problem
    test_ei = ac_model._compute_acq(test_x[:, np.newaxis])
    best_ei_score = np.max(test_ei)
    best_ei_x = test_x[np.argmax(test_ei)]

    # Define plotting function
        # Get lower and upper confidence bounds
        # lower, upper = rand_var.confidence_region()
        # Visualization training data as black stars
    pre_mean = observed_pred_y
    pre_best_y = np.min(pre_mean)
    pre_best_x = test_x[np.argmin(pre_mean)]
    pre_up = observed_pred_y + observed_corv
    pre_low = observed_pred_y - observed_corv

    ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
    ax.plot(test_x, pre_mean[:,0], 'b-', linewidth=1, alpha=1)
    ax.plot(test_x, test_ei[:,0], 'g-', linewidth=1, alpha=1)

    ax.plot(train_x[:,0], train_y_temp[:,0], marker='*', color='black', linewidth=0)
    # ax[0].plot(Ac_candi[:,0], Ac_candi_f[:,0], marker='*', color='orange', linewidth=0)
    ax.plot(Ac_candi[:, 0], 0, marker='*', color='orange', linewidth=0)
    # ax.plot(best_ei_x, best_ei_score, marker='*', color='green', linewidth=0)
    # ax[0].plot(opt_x[:,0], opt_val, marker='*', color='red', linewidth=0)
    ax.plot(opt_x, 0, marker='*', color='red', linewidth=0)
    ax.plot(pre_best_x, pre_best_y, marker='*', color='blue', linewidth=0)
    ax.fill_between(test_x, pre_up[:,0], pre_low[:,0], alpha=0.2, facecolor='blue')

    if not os.path.exists('{}/figs/oneD/{}/{}'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}')):
        os.makedirs('{}/figs/oneD/{}/{}'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}'))

    if conformal and model.qhats is not None:
        ax.plot(test_x, (observed_pred_y + cfv)[:, 0], linestyle='--',  color='purple')
        ax.plot(test_x, (observed_pred_y - cfv)[:, 0], linestyle='--', label='cp naive', color='purple')
        np.savetxt('{}/toy/{}/{}/{}_ei.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                   file_name), np.concatenate((test_x[:, np.newaxis], test_ei), axis=1))
    ax.legend()

    #
    # ax[0].legend(['True f(x)', 'Prediction', 'EI', 'Observed Data', 'Candidate'])
    # ax[0].set_xlim([bounds[0][0], bounds[1][0]])
    # num_sample = train_x.shape[0]
    # ax[0].set_title(title + ' at Seed=' + str(Seed) + ' Sample(' + str(num_sample) + ')')
    #
    # # plt.subplot(grid[0, 2])
    # ax[1].text(1, 1, "Prediction:\n"
    #                  "x={:.4}, y={:.4}\n"
    #                  "\n"
    #                  "True f(x):\n"
    #                  "x={:.4}, y={:.4}\n"
    #                  "\n"
    #                  "EI:\n"
    #                  "x={:.4}, y={:.4}\n"
    #                  "\n"
    #                  "Candidate:\n"
    #                  "x={:.4}, y={:.4}".format(
    #     pre_best_x,
    #     pre_best_y,
    #     opt_x[0][0],
    #     opt_val,
    #     best_ei_x,
    #     best_ei_score,
    #     Ac_candi[0][0],
    #     Ac_candi_f[0][0]
    # ), fontsize=12)
    #
    # ax[1].axis([0, 10, 0, 10])
    # ax[1].axis('off')

    plt.grid()



    plt.savefig('{}/figs/oneD/{}/{}/{}.png'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}', file_name), format='png')

    os.makedirs('{}/toy/{}/{}/'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}'), exist_ok=True)
    np.savetxt('{}/toy/{}/{}/{}_true.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                   file_name), np.concatenate((test_x[:, np.newaxis], test_y[:, np.newaxis]), axis=1))
    np.savetxt('{}/toy/{}/{}/{}_pred_y.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                   file_name), np.concatenate((test_x[:, np.newaxis], observed_pred_y), axis=1))
    np.savetxt('{}/toy/{}/{}/{}_cov_lower.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                   file_name),np.concatenate((test_x[:, np.newaxis], observed_pred_y - observed_corv), axis=1))
    np.savetxt('{}/toy/{}/{}/{}_cov_higher.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                   file_name),np.concatenate((test_x[:, np.newaxis], observed_pred_y + observed_corv), axis=1))
    np.savetxt('{}/toy/{}/{}/{}_ei.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                   file_name), np.concatenate((test_x[:, np.newaxis], test_ei), axis=1))

    np.savetxt('{}/toy/{}/{}/{}_train.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                   file_name), np.concatenate((train_x, train_y_temp), axis=1))

    if conformal and model.qhats is not None:
        np.savetxt(
            '{}/toy/{}/{}/{}_cfv_lower.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                   file_name),
            np.concatenate((test_x[:, np.newaxis], observed_pred_y - cfv), axis=1))
        np.savetxt(
            '{}/toy/{}/{}/{}_cfv_higher.txt'.format(Exper_floder, model.model_name, f'{Env.get_current_task_name()}',
                                                    file_name),
            np.concatenate((test_x[:, np.newaxis], observed_pred_y + cfv), axis=1))
    plt.close()



if __name__ == '__main__':
    Task_list = [
        'Sphere',
        'Ackley',
        'Griewank',
        'Levy',
        'StyblinskiTang',
        ]



