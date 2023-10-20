import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


class CNN(nn.Module):
    def __init__(self, kernel_size=3, pool_size=2, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * ((32 // pool_size) // pool_size) ** 2, 128)  # Adjusted for variable pool_size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HPOCNN:
    def __init__(self, name, task_id, Seed=0):
        np.random.seed(Seed)
        torch.manual_seed(Seed)
        self.name = name
        self.task_type = 'Continuous'
        self.task_id = task_id
        dataset_name = ['svhn', 'cifar10', 'cifar100']
        self.dataset_name = dataset_name[task_id]
        self.Variable_range = [[0, 1], [-10, 0], [-10, -5], [2, 5]]  # Added range for kernel_size and pool_size
        self.Variable_type = ['float', 'float', 'float', 'int']  # kernel_size and pool_size are int
        self.Variable_name = ['momentum', 'learning_rate', 'weight_decay', 'kernel_size']  # Added new variable names
        self.log_flag = [False, True, True, False]
        self.bounds = np.array([[-1.0] * len(self.Variable_range), [1.0] * len(self.Variable_range)])
        self.RX = np.array(self.Variable_range)
        self.query_num = 0

    def transfer(self, X):
        X = np.array(X)
        X = (X + 1) * (self.RX[:, 1] - self.RX[:, 0]) / 2 + (self.RX[:, 0])
        idx_list = [idx for idx, i in enumerate(self.log_flag) if i is True]
        X[:, idx_list] = np.exp2(X[:, idx_list])
        int_flag = [idx for idx, i in enumerate(self.Variable_type) if i == 'int']
        X[:, int_flag] = X[:, int_flag].astype(int)
        return X

    def load_data(self, batch_size=64):
        # similar to the original load_data Method but adapted for SimpleCNN
        dataset_name = self.dataset_name
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if dataset_name == 'svhn':
            trainset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
            testset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        elif dataset_name == 'cifar10':
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'cifar100':
            trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset_name))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        if dataset_name == 'svhn' or dataset_name == 'cifar10':
            n_outputs = 10
        elif dataset_name == 'cifar100':
            n_outputs = 100

        return trainloader, testloader, n_outputs

    def get_score(self, configuration: dict):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.set_device(1)
        else:
            device = torch.device('cpu')

        epochs = 30
        batch_size = 64
        lr = configuration['lr']
        momentum = configuration['momentum']
        weight_decay = configuration['weight_decay']
        kernel_size = int(configuration['kernel_size'])


        trainloader, testloader, n_output = self.load_data(batch_size=batch_size)
        net = CNN(kernel_size=kernel_size, num_classes=n_output).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print('Epoch %d, Loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
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
                'momentum': x[0],
                'lr': x[1],
                'weight_decay':x[2],
                'kernel_size':x[3],
                'batch_size': 64
            }
            Y.append(1 - self.get_score(configuration))
        return np.array(Y)
if __name__ == '__main__':
    mlp = HPOCNN(name='Res',task_id=0,Seed=0)
    mlp.f([[-0.1, 0.4, 0.9,-0.1]])