import matplotlib.pyplot as plt
import numpy as np


def get_max_values(data):
    max_values = [max(data[:i+1]) for i in range(len(data))]
    return max_values

data_list = [0.575,0.169,0.576,0.432,0.431,0.432,0.432,0.575,0.162,0.167,0.431,0.573,0.571,
             0.168,0.163,0.430,0.577,0.578,0.576,0.168,0.431,0.442,0.433,0.162,0.157,0.158,
             0.577,0.579,0.578,0.579,0.433,]

max_values = get_max_values(data_list)

# plt.plot(np.arange(len(data_list)), max_values, marker='o')
# plt.axvline(x=12,color='red')
# plt.xlabel('Function evaluation')
# plt.ylabel('Accuracy')
# plt.title('Transformer optimization trajctory')
# plt.grid(True)
# plt.savefig('./traj_tranf')

plt.hist(data_list)
plt.savefig('./hist')
