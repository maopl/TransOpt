import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms



class HPOMLP:
    def __init__(self, name, task_id, Seed=0):
        np.random.seed(Seed)
        torch.manual_seed(Seed)
        self.seed = Seed
        self.name = name
        self.n_var = 4
        self.task_type = 'Continuous'

        dataset_name = ['mnist', 'cifar10', 'cifar100']
        self.dataset_name = dataset_name[task_id]

        self.query_num = 0

        self.Variable_range = [[2,8], [0,1], [-10,0], [0,1]]
        self.Variable_type = ['int', 'float', 'float', 'float']
        self.Variable_name = ['width', 'momentum', 'lr', 'activate_weights']
        self.log_flag = [True, False, True, False]

        self.bounds = np.array([[-1.0] * self.n_var, [1.0] * self.n_var])

        self.RX = np.array(self.Variable_range)

        # self.space = []
        # eps = 1e-6
        # for i in range(self.n_var):
        #     self.space.append(ContinuousParameter(f"x{i + 1}", 0, 1 - eps))
        # self.space = ParameterSpace(self.space)
        #

    def transfer(self, X):
        X = np.array(X)
        X = (X + 1) * (self.RX[:, 1] - self.RX[:, 0]) / 2 + (self.RX[:, 0])
        idx_list = [idx for idx, i in enumerate(self.log_flag) if i is True]
        X[:, idx_list] = np.exp2(X[:, idx_list])
        int_flag = [idx for idx, i in enumerate(self.Variable_type) if i == 'int']
        X[:, int_flag] = X[:, int_flag].astype(int)
        return X

    def normalize(self, X):
        return 2 * (X - (self.RX[:, 0])) / (self.RX[:, 1] - self.RX[:, 0]) - 1

    def load_data(self, batch_size=64):
        dataset_name = self.dataset_name
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if dataset_name == 'mnist':
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset_name))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        if dataset_name == 'mnist':
            n_inputs = 784
            n_outputs = 10
        elif dataset_name == 'cifar10':
            n_inputs = 3072
            n_outputs = 10
        elif dataset_name == 'cifar100':
            n_inputs = 3072
            n_outputs = 100

        return trainloader, testloader, n_inputs, n_outputs

    def get_score(self, configuration: dict):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.device_count()
            torch.cuda.set_device(1)
        else:
            device = torch.device('cpu')
        epochs = 30
        batch_size = 64

        n_layers = 2
        n_neurons = int(configuration['width'])
        lr = configuration['lr']
        momentum = configuration['momentum']
        activate_weights = configuration['activate_weights'] * np.ones((n_layers,))

        trainloader, testloader, n_input, n_output = self.load_data(batch_size=batch_size)

        net = Net(n_input, n_output, n_layers, n_neurons, activate_weights).to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        for e in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.view(inputs.size(0), -1)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print('Epoch %d, Loss: %.3f' % (e + 1, running_loss / len(trainloader)))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                images = images.view(images.size(0), -1)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print('Accuracy: %.2f %%' % (100 * accuracy))

        return accuracy

    def f(self, X):
        self.query_num += 1
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        Y = []
        for x in X:
            configuration = {
                'width': x[0],
                'momentum': x[1],
                'lr': x[2],
                'activate_weights': x[3],
                'batch_size': 64
            }
            Y.append(1 - self.get_score(configuration))
        return np.array(Y)

class WeightedSigmoidSeedReLU(torch.nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return self.weight * torch.relu(x) + (1 - self.weight) * torch.sigmoid(x)


class MLP(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hiddens, weights):
        super().__init__()
        layers = []

        sizes = [n_inputs] + n_hiddens + [n_outputs]

        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(WeightedSigmoidSeedReLU(weights[i]))
        layers.append(torch.nn.LogSoftmax(dim=1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_layers,
                 n_eurons,
                 weights):
        super().__init__()
        n_hiddens = [n_eurons] * n_layers
        self.net = MLP(n_inputs, n_outputs, n_hiddens, weights)

    def forward(self, x):
        output = self.net(x)
        return output

if __name__ == '__main__':
    mlp = HPOMLP('MLP', 0)
    mlp.f([[0.5, -0.3, 0.1, -0.1]])