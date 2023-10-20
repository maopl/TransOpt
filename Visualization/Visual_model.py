import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from matplotlib import cm
import warnings


def Visual_task_recognition(file_name, target_model, target_X, target_Y,
                            source_model_list, source_X_list, source_Y_list, WD_list,
                            model_name_list, Exper_floder=None, dtype=np.float64):

    fig = plt.figure(figsize=(12, 8))
    grid = GridSpec(1, 4, figure=fig)

    # Test points every 0.02 in [0,1]
    test_x = np.arange(-1, 1.05, 0.005, dtype=dtype)
    model_num = len(source_model_list)
    legend = []
    legend_text = []

    ax1 =  fig.add_subplot(grid[0, 0:3])

    ###Plot target model
    target_y, target_corv = target_model.predict(test_x[:, np.newaxis])

    pre_mean = target_y
    pre_up = target_y + target_corv
    pre_low = target_y - target_corv

    l1, = ax1.plot(test_x, pre_mean[:, 0], 'r', linewidth=1, alpha=1)
    legend.append(l1)
    legend_text.append(f'target model')

    p1, = ax1.plot(target_X[:, 0], target_Y[:, 0], marker='*', color='black', linewidth=0)
    legend.append(p1)
    legend_text.append(f'target observed point')
    # ax[0].plot(opt_x[:,0], opt_val, marker='*', color='red', linewidth=0)
    ax1.fill_between(test_x, pre_up[:, 0], pre_low[:, 0], alpha=0.2, facecolor='red')

    for i in range(model_num):
        pred_y, corv = source_model_list[i].predict(test_x[:, np.newaxis])

        pre_mean = pred_y
        pre_up = pred_y + corv
        pre_low = pred_y - corv

        l, = ax1.plot(test_x, pre_mean[:, 0], 'b-', linewidth=1, alpha=1)
        # legend.append(l)
        # legend_text.append(f'source model {model_name_list[i]}')

        ax1.fill_between(test_x, pre_up[:, 0], pre_low[:, 0], alpha=0.2, facecolor='blue')

    ax1.legend(handles=legend, labels=legend_text)
    ax1.set_xlim([-1, 1])
    # num_sample = X.shape[0]
    # ax[0].set_title(title + ' at Seed=' + str(Seed) + ' Sample(' + str(num_sample) + ')')

    text = [f"WD to {model_name_list[i]} :{WD_list[i]} \n" for i in range(model_num)]
    text = ' '.join(text)
    ax2 = fig.add_subplot(grid[0, 3])
    ax2.axis([0, 1, 0, 1])
    ax2.text(-0.1, 0.8, text, fontsize=12)
    ax2.axis('off')

    # plt.show()

    if not os.path.exists('{}/figs/task_recognition'.format(Exper_floder)):
        os.makedirs('{}/figs/task_recognition'.format(Exper_floder))

    plt.savefig('{}/figs/task_recognition/{}.png'.format(Exper_floder, file_name), format='png')
    plt.close()