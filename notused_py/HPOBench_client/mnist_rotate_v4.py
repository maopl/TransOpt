import numpy as np
from PIL import Image
import torch
from time import time
from functools import partial
from emukit.core import ParameterSpace, ContinuousParameter
import copy


class MnistRotateV4:
    def __init__(self, name, Xdim, rot, Seed=0, bounds = None):
        np.random.seed(Seed)
        torch.manual_seed(Seed)
        self.name = name
        self.rot = rot
        self.n_var = Xdim

        self.query_num = 0

        self.Variable_range = [[0,1], [0,1], [0,1], [0,1]]
        self.Variable_type = ['float', 'float', 'float', 'float']
        self.Variable_name = ['momentum', 'n_neurals', 'learning_rate', 'activate_weights']
        if bounds is None:
            self.bounds = np.array([[-1.0] * self.n_var, [1.0] * self.n_var])
        else:
            self.bounds = bounds
        self.RX = np.array(self.Variable_range)

        self.space = []
        self.function = partial(MnistRotateV4.mnist_rotate_function, rot=self.rot, init_seed=Seed)
        eps = 1e-6
        for i in range(self.n_var):
            self.space.append(ContinuousParameter(f"x{i + 1}", 0, 1 - eps))
        self.space = ParameterSpace(self.space)

    def transfer(self, X):
        return (X + 1) * (self.RX[:, 1] - self.RX[:, 0]) / 2 + (self.RX[:, 0])

    def normalize(self, X):
        return 2 * (X - (self.RX[:, 0])) / (self.RX[:, 1] - self.RX[:, 0]) - 1

    @staticmethod
    def rotate_dataset(data, rotation):
        output = data.copy()
        for i in range(len(data)):
            img = Image.fromarray(data[i], mode='L')
            img_rotate = img.rotate(rotation)
            output[i] = np.array(img_rotate)
        return output

    @staticmethod
    def load_data(rot):
        raw_file = np.load('./Bench/HPOBench/Data/mnist.npz')
        x_train = raw_file['x_train']
        y_train = raw_file['y_train']
        x_test = raw_file['x_test']
        y_test = raw_file['y_test']
        x_train_rot = MnistRotateV4.rotate_dataset(x_train, rot)
        x_test_rot = MnistRotateV4.rotate_dataset(x_test, rot)

        x_train_rot = torch.tensor(x_train_rot, dtype=torch.float).view(x_train_rot.shape[0], 784)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_test_rot = torch.tensor(x_test_rot, dtype=torch.float).view(x_test_rot.shape[0], 784)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return x_train_rot, y_train, x_test_rot, y_test

    @staticmethod
    def mnist_rotate_function(xs, rot, output_noise: float = 0.0, init_seed=0):
        epochs = 30
        batch = 100
        retry_threshold = 0.5
        retry_maximum = 0
        xs = np.atleast_2d(xs)
        n_sample, n_var = xs.shape
        if n_var != 4:
            print("Error: n_var is not 4!")
            exit()
        y = np.zeros(shape=(n_sample, 1))
        for idx in range(n_sample):
            '''
            x[0]: momentum [0,1]
            x[1]: n_neurals [16,32,48,64,80,96,112,128]
            x[2]: learning_rate [0.0001, 0.05]
            x[3]: activate_weights [0,1]
            '''
            n_layer = 2
            n_neurals = 16 * (int(xs[idx, 1] * 8) + 1) * np.ones((n_layer,),dtype=int)
            lr = xs[idx, 2] * 0.0499 + 0.0001
            momentum = xs[idx, 0]
            seed = 1
            w = xs[idx, 3] * np.ones((n_layer,))
            print('HPOBench logger: xs', xs[idx, ], '%d/%d'%(idx+1,n_sample))
            print('HPOBench logger: lr %.4f, neural %d, w %.2lf, momentum %.2lf'%(lr, n_neurals[0] ,xs[idx, 3],momentum))
            x_train, y_train, x_test, y_test = MnistRotateV4.load_data(rot)

            time0 = time()
            while True:
                net = copy.deepcopy(MnistRotateV4.Net(784, 10, n_neurals, lr, momentum, w, seed+init_seed))
                for e in range(epochs):
                    fr = 0
                    to = batch
                    running_loss = 0.0
                    while to <= x_train.shape[0]:
                        loss = net.observe(x_train[fr:to], y_train[fr:to])
                        running_loss += loss
                        fr += batch
                        to += batch
                    print('\rHPOBench logger: epoch %d loss %.2f %%' % (e, running_loss/x_train.shape[0]*batch),  end="\b")
                print()
                y[idx, 0] = 1 - MnistRotateV4.eval_tasks(net, x_test, y_test).item()
                if y[idx, 0] < retry_threshold or seed >= retry_maximum:
                    break
                seed += 1
                print('HPOBench logger: fail to fit (loss %.2f %%), retry with another seed (%d)'%(y[idx, 0]*100, seed))
            print('HPOBench logger: FE %d s, loss %.2f %%' % (int(time() - time0),y[idx, 0]*100))
        return y[:,0]

    def f(self, X):
        self.query_num += 1
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        return self.function.func(X, self.rot)

    @staticmethod
    def eval_tasks(model, x_test, y_test):
        model.eval()
        _, pb = torch.max(model(x_test), 1, keepdim=False)
        rt = (pb == y_test).float().sum()
        return rt / x_test.size(0)

    class WeightedSigmoidSeedReLU(torch.nn.Module):
        def __init__(self, weight=1):
            super().__init__()
            self.weight = weight

        def forward(self, x):
            return self.weight * torch.relu(x) + (1 - self.weight) * torch.sigmoid(x)

    class MLP(torch.nn.Module):
        def __init__(self, sizes, weights):
            super(MnistRotateV4.MLP, self).__init__()
            layers = []

            for i in range(0, len(sizes) - 1):
                layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
                if i < (len(sizes) - 2):
                    layers.append(MnistRotateV4.WeightedSigmoidSeedReLU(weights[i]))
            layers.append(torch.nn.LogSoftmax(dim=1))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class Net(torch.nn.Module):

        def __init__(self,
                     n_inputs,
                     n_outputs,
                     n_hiddens,
                     lr, momentum, weights, seed):
            super(MnistRotateV4.Net, self).__init__()

            self.net = MnistRotateV4.MLP([n_inputs] + list(n_hiddens) + [n_outputs], weights)

            # setup optimizer

            self.opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

            # setup losses
            self.bce = torch.nn.NLLLoss()
            self.nc_per_task = n_outputs
            self.n_outputs = n_outputs
            self.seed = seed

        def forward(self, x):
            output = self.net(x)
            return output

        def observe(self, x, y):
            torch.random.manual_seed(self.seed)
            self.train()
            self.zero_grad()
            out = self(x)
            loss = self.bce(out, y)
            loss.backward()
            self.opt.step()
            return loss.item()



if __name__ == '__main__':
    bench = MnistRotateV4(4)
    f = bench.functions[0]
    print(f.func([[0.1,0.2,0.5,0.5],[0.3,0.2,0.3,0.1]], bench.rots[1]))