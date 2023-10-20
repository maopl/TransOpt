import numpy as np
from Util.Normalization import Normalize


def traj_metric(Y:np.ndarray, initial_size):
    Y = Normalize(Y)

    best_y = np.min(Y[:initial_size])
    truncated_Y = Y[initial_size:]

    promotion = []

    new_id = 0
    for j in range(len(truncated_Y)):
        new_Y = np.min(truncated_Y[:j+1])

        if new_Y < best_y:
            promotion.append(np.square(best_y - new_Y)/ (j - new_id + 1))
            new_id = j

    perf = np.sqrt(np.sum(np.array(promotion)))

    return perf



def trend_metric(X:np.ndarray, Y:np.ndarray):
    Y = Normalize(Y)

    worst_id = np.argmax(Y)

    worst_y = Y[worst_id]
    worst_x = X[worst_id]

    slop = []
    for i in range(X.shape[1]):
        slop_d = 0
        for j in range(len(Y)):
            if j == worst_id:
                continue
            slop_d += np.abs(worst_y - Y[j]) / np.abs(worst_x[i] - X[j][i])
        slop.append(slop_d/(len(Y) - 1))

    return np.array(slop)

def walk(X:np.ndarray, step_num, repeat_num):
    n, m = X.shape
    G = np.dot(X, X.T)
    D = np.zeros([n,n])
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = np.sqrt(G[i,i] - 2 * G[i, j] + G[j, j])
            D[j, i] = D[i,j]

    step_list = []
    for r in range(repeat_num):
        start_id = np.random.randint(0,n)
        dis_vec = D[start_id]

        step_list.append(np.argsort(dis_vec)[:step_num])

    return np.array(step_list)



def rug_metric(X:np.ndarray, Y:np.ndarray):
    Y = Normalize(Y)

    walk_step = walk(X, 5, 10)
    rug_perf = []

    for d in range(X.shape[1]):
        slope = 0
        for w in walk_step:
            y_diff = 0
            for p_id in range(len(w) - 1):
                y_diff += np.square(np.abs(Y[p_id + 1] - Y[p_id]) / np.abs(X[p_id + 1][d] - X[p_id][d]))
            y_diff = np.sqrt(y_diff / (len(w) - 1))
            slope += y_diff
        rug_perf.append(slope/len(walk_step))

    return np.array(rug_perf)





