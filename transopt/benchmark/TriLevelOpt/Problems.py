import argparse
import datetime
import json
import numpy as np
import os
import time
import math
import sys
from typing import Iterable
import glob
import shutil
from pathlib import Path
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from betty.utils import log_from_loss_dict
from betty.logging import logger
import wandb

import torch.nn.functional as F

from transopt.benchmark.TriLevelOpt.models import Hyperparameters
        

class NetworkTraining(ImplicitProblem):
    def __init__(self, name, config, module=None, optimizer=None,
                 scheduler=None, train_data_loader=None, extra_config=None):
        super().__init__(name, config, module, optimizer, scheduler,
                        train_data_loader, extra_config)
        if torch.cuda.is_available():
            gpu_id = extra_config.gpu_id if hasattr(extra_config, 'gpu_id') else 0
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
            
        # Initialize wandb monitoring
        wandb.init(
            project="network_training",
            config={
                "architecture": "WideResNet",
                "device": str(self.device)
            }
        )
        
    def training_step(self, batch, hparams):
        # Get hyperparameters from outer problem
        X, Y = batch
        all_x = torch.stack([x for x in X]).to(self.device)
        all_y = torch.stack([y for y in Y]).to(self.device)

        # Forward pass
        predictions = self.module(all_x, hparams)

        ce_loss = F.cross_entropy(predictions, all_y)

        loss = ce_loss

        correct = (predictions.argmax(1) == all_y).sum().item()
        total = all_y.size(0)
        accuracy = correct / total

        # Enhanced wandb logging
        wandb.log({
            'train_loss': loss.item(),
            'train_accuracy': accuracy,
            'train_correct': correct,
            'train_total': total,
            'cross_entropy_loss': ce_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch_size': total,
        })

        return loss
    
    def configure_optimizer(self):
        optimizer = torch.optim.SGD(self.module.parameters())
        # Log optimizer config
        wandb.config.update({
            "optimizer": "SGD",
            "initial_lr": optimizer.param_groups[0]['lr']
        })
        return optimizer

    def on_inner_loop_start(self):
        # Reset parameters at start of inner loop
        for p in self.module.parameters():
            p.data.zero_()
            
        # Log parameter statistics
        total_params = sum(p.numel() for p in self.module.parameters())
        wandb.log({
            "total_parameters": total_params,
            "parameter_norm": sum(p.norm().item() for p in self.module.parameters())
        })

    
class HyperparameterOptimization(ImplicitProblem):
    def training_step(self, batch):
        # Unroll inner problem
        x, target = batch
        hparams = self.forward()
        loss = self.network_training(x, hparams, target)

        return loss


class DataTuning(ImplicitProblem):
    def __init__(self, name, config, module=None, optimizer=None,
                 scheduler=None, train_data_loader=None, extra_config=None):
        super().__init__(name, config, module, optimizer, scheduler,
                        train_data_loader, extra_config)
        pass
    
    def training_step(self, batch):
        X, Y = batch
        hparams = self.forward()
        loss = self.network_training(X, hparams, Y)
        return loss
    
    def forward(self):
        return self.lr, self.weight_decay, self.momentum, self.batch_size, self.dropout_rate

