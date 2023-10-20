import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_change_detection(Y:np.ndarray, true_partition, R_pdf):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    total_num = 0
    length = len(Y)
    previous_totalnum = 0
    for pid, i in enumerate(true_partition):
        total_num += true_partition[pid]
        ax[0].scatter(list(range(previous_totalnum, total_num)), Y[previous_totalnum:total_num])
        previous_totalnum = total_num
    sparsity = 1
    epsilon = 1e-7
    density_matrix = -np.log(R_pdf[0:length:sparsity, 0:length:sparsity] + epsilon)
    ax[1].pcolor(np.array(range(0, len(R_pdf[:length, 0]), sparsity)),
                 np.array(range(0, len(R_pdf[:length, 0]), sparsity)),
                 density_matrix,
                 cmap=cm.Greys, vmin=0, vmax=density_matrix.max(),
                 shading='auto')
    plt.savefig('change detection')




if __name__ == '__main__':
    Y = np.random.random(20)
    plot_change_detection(Y, [5,6,2,7],2)