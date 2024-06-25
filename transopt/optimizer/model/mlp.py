import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets.utils as dataset_utils
from PIL import Image
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from transopt.agent.registry import model_registry
from transopt.optimizer.model.model_base import Model

def compute_irm_penalty(losses, dummy):
    g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
    g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
    return (g1 * g2).sum()


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__(input_dim=input_dim)
        self.fc1 = nn.Linear(input_dim, 512)  # 输入维度改为10
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)  # 输出维度改为1

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    


@model_registry.register('MLP')
class MLP(Model):
    def __init__(self, config):
        super().__init__()
        self._model = None
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
    
    def meta_fit(
        self,
        source_X : List[np.ndarray],
        source_Y : List[np.ndarray],
        **kwargs,
    ):
        pass

    def fit(
        self,
        X : np.ndarray,
        Y : np.ndarray,
        epoch : int = 30,
        optimize: bool = False,
    ):
        self._X = np.copy(X)
        self._y = np.copy(Y)
        self._Y = np.copy(Y)

        _X = np.copy(self._X)
        _y = np.copy(self._y)
        
        
        X_tensor = torch.tensor(_X, dtype=torch.float32)
        y_tensor = torch.tensor(_y, dtype=torch.float32).view(-1, 1)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        
        if self._model is None:
            self._model = Net(input_dim=_X.shape[1]).to(self.device)
            optimizer = optim.Adam(self._model.parameters(), lr=0.001)


        self._model.train()
        train_loaders = [iter(x) for x in train_loaders]
        dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(self.device)

        batch_idx = 0
        # penalty_multiplier = epoch ** 1.6
        penalty_multiplier = 0
        print(f'Using penalty multiplier {penalty_multiplier}')
        while True:
            optimizer.zero_grad()
            error = 0
            penalty = 0
            for loader in train_loaders:
                data, target = next(loader, (None, None))
                if data is None:
                    return
                data, target = data.to(self.device), target.to(self.device).float()
                output = self._model(data)
                loss_erm = F.mse_loss(output * dummy_w, target, reduction='none')
                penalty += compute_irm_penalty(loss_erm, dummy_w)
                error += loss_erm.mean()
            (error + penalty_multiplier * penalty).backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loaders[0]),
                    100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
                # print('First 20 logits', output.data.cpu().numpy()[:20])

            batch_idx += 1
        
    def predict(
        self, X: np.ndarray, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        self._model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self._model(data)
        output = output.to('cpu')
        output = output.numpy()
        variance = np.zeros(shape=(output.shape[0], 1))
        return output, variance
        
