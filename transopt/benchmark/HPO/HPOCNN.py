import logging
import time
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.optimizer.sampler.random import RandomSampler
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.utils.openml_data_manager import OpenMLHoldoutDataManager


class BGRed(object):

    def __call__(self, img):
        img = np.array(img)
        dtype = img.dtype
        h, w = img.shape
        img = np.reshape(img, [h, w, 1])
        
        # Convert to red
        img = np.concatenate([img, np.zeros((h, w, 2), dtype=dtype)], axis=2)
        # Convert to green
        # img = np.concatenate([np.zeros((h, w, 1), dtype=dtype), img, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    
        
        # images[i, 0, :, :] = 0  # Green
        return img  # Returning the first image as an example


class BGGreen(object):

    def __call__(self, img):
        img = np.array(img)
        dtype = img.dtype
        h, w = img.shape
        img = np.reshape(img, [h, w, 1])
        
        # Convert to red
        # img = np.concatenate([img, np.zeros((h, w, 2), dtype=dtype)], axis=2)
        # Convert to green
        img = np.concatenate([np.zeros((h, w, 1), dtype=dtype), img, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    
        return img  # Returning the first image as an example
    
class BGBlue(object):

    def __call__(self, img):
        img = np.array(img)
        dtype = img.dtype
        h, w = img.shape
        img = np.reshape(img, [h, w, 1])
        
        # Convert to blue
        img = np.concatenate([np.zeros((h, w, 2), dtype=dtype), img], axis=2)
    
        return img  # Returning the first image as an example



transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)



    
class Learner(nn.Module):
    in_channels: int = 3
    conv1_filters = 64
    conv2_filters = 128
    # learner_fc_size: int = 64  # not used

    mp1_size: int = 2
    mp2_size: int = 2
    input_size: int = 28
    leaky_relu_alpha: float = 0.1
    batch_norm_momentum: float = 0.1  # should be 0.0 as written in paper, but it reduces the performance

    def __init__(self, target_classes):
        super().__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        self.kernel_size = 3
        self.in_channels = 3
        self.target_classes = target_classes
        
        self.conv1 = nn.Conv2d(self.in_channels, self.conv1_filters, self.kernel_size, 1)
        self.bn1 = nn.BatchNorm2d(self.conv1_filters, momentum=self.batch_norm_momentum)
        self.conv2 = nn.Conv2d(self.conv1_filters, self.conv2_filters, self.kernel_size, 1)
        self.bn2 = nn.BatchNorm2d(self.conv2_filters, momentum=self.batch_norm_momentum)
        c1_size = (self.input_size - self.kernel_size + 1) // self.mp1_size
        c2_size = (c1_size - self.kernel_size + 1) // self.mp2_size
        self.fc = nn.Linear(self.conv2_filters * c2_size * c2_size, self.target_classes)
        nn.init.kaiming_normal_(self.fc.weight, self.leaky_relu_alpha)
        self.bn_fc = nn.BatchNorm1d(self.target_classes, momentum=self.batch_norm_momentum)
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.bn1(x)
        x = F.max_pool2d(x, self.mp1_size)

        x = self.conv2(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.bn2(x)
        x = F.max_pool2d(x, self.mp2_size)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn_fc(x)

        output = F.log_softmax(x, dim=1)
        return output


@problem_registry.register("CNN")
class HPOCNN(NonTabularProblem):
    DATASET_NAME = ["svhn", "cifar10", "cifar100", "minist", "colored_minist"]
    problem_type = 'hpo'
    num_variables = 10
    num_objectives = 1
    workloads = []
    fidelity = None
    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        super(HPOCNN, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset_name = HPOCNN.DATASET_NAME[workload]


    def get_configuration_space(
            self, seed: Union[int, None] = None):
            """
            Creates a ConfigSpace.ConfigurationSpace containing all parameters for
            the XGBoost Model

            Parameters
            ----------
            seed : int, None
                Fixing the seed for the ConfigSpace.ConfigurationSpace

            Returns
            -------
            ConfigSpace.ConfigurationSpace
            """
            variables=[Continuous('lr', [-10.0, 0.0]),
                    Continuous('momentum', [0.0, 1.0]),
                    Continuous('weight_decay', [-10.0, -5.0]),
                    ]
            ss = SearchSpace(variables)
            return ss
        
    def get_fidelity_space(
        self, seed: Union[int, None] = None):
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """

        # return fidel_space
        fs = FidelitySpace([])
        return fs
    def load_data(self, batch_size=64):
        dataset_name = self.dataset_name

        if dataset_name == "svhn":
            trainset = datasets.SVHN(
                root="./data", split="train", download=True, transform=transform
            )
            testset = datasets.SVHN(
                root="./data", split="test", download=True, transform=transform
            )
        elif dataset_name == "cifar10":
            trainset = datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            testset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "cifar100":
            trainset = datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transform
            )
            testset = datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "minist":
            trainset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )
            testset = datasets.MNIST(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "colored_minist":
            trainset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transforms.Compose(
                    [
                        BGRed(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            )
            testset = datasets.MNIST(
                root="./data", train=False, download=True, transform=transforms.Compose(
                    [
                        BGGreen(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            )
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )

        if dataset_name == "svhn" or dataset_name == "cifar10" or dataset_name == "minist" or dataset_name == "colored_minist":
            n_outputs = 10
        elif dataset_name == "cifar100":
            n_outputs = 100

        return trainloader, testloader, n_outputs
    
    def get_score(self, configuration: dict):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_device(1)
        else:
            device = torch.device("cpu")
        epochs = 30
        batch_size = 64

        lr = configuration["lr"]
        momentum = configuration["momentum"]
        weight_decay = configuration["weight_decay"]

        trainloader, testloader, n_output = self.load_data(batch_size=batch_size)

        net = Learner(target_classes=n_output).to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay = weight_decay,
        )
        start_time = time.time()
        for e in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print("Epoch %d, Loss: %.3f" % (e + 1, running_loss / len(trainloader)))

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
        end_time = time.time()
        print("Accuracy: %.2f %%" % (100 * accuracy))

        return accuracy, end_time - start_time



    def objective_function(
        self,
        configuration,
        fidelity = None,
        seed = None,
        **kwargs
    ) -> Dict:
        if fidelity is None:
            fidelity = {"epoch": 30, "data_frac": 0.8}
        c = {
            "momentum": configuration["momentum"],
            "lr": np.exp2(configuration["lr"]),
            "weight_decay": np.exp2(configuration["weight_decay"]),
            "batch_size": 64,
            "epoch": fidelity["epoch"],
        }
        acc, time = self.get_score(c)

        results = {list(self.objective_info.keys())[0]: float(1 - acc)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results
    
    def get_objectives(self) -> Dict:
        return {'function_value': 'minimize'}
    
    def get_problem_type(self):
        return "hpo"

if __name__ == "__main__":
    CNN = HPOCNN(
        task_name="Res", budget_type='FEs', budget=40, seed=0, task_type="non-tabular", workload=4
    )
    configuration = {
        "momentum": 0.1,
        "lr": -0.3,
        "weight_decay": -5,
    }
    CNN.f(configuration=configuration)
