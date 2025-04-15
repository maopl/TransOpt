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
import wandb

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

from transopt.benchmark.TriLevelOpt.Problems import NetworkTraining, HyperparameterOptimization, DataTuning
from transopt.benchmark.TriLevelOpt.models import Hyperparameters
from transopt.benchmark.TriLevelOpt.model_search import Network, DataParameters
from transopt.benchmark.TriLevelOpt.primitives import sub_policies

from transopt.benchmark.HPO import datasets

parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--base_dir", type=str, default="~", help="location of the data corpus"
)

parser.add_argument(
    "--darts_type", type=str, default="PCDARTS", help="[DARTS, PCDARTS]"
)
parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--init_ch", type=int, default=16, help="num of init channels")
parser.add_argument("--report_freq", type=int, default=100, help="report frequency")
parser.add_argument("--tuning_steps", type=int, default=4, help="data tuning steps")
parser.add_argument("--unroll_steps", type=int, default=1, help="unrolling steps")

args = parser.parse_args()

dataset = vars(datasets)['RobCifar10'](root=None, augment=None)


hyperparameters = Hyperparameters()
hyper_optimizer = torch.optim.Adam(
    hyperparameters.parameters(),
    lr=0.001,
    betas=(0.5, 0.999),
    weight_decay=0.0001,
)

data_net = DataParameters(sub_policies=sub_policies)
data_optimizer = torch.optim.Adam(
    data_net.parameters(),
    lr=0.001,
    betas=(0.5, 0.999),
    weight_decay=0.0001,
)


model = Network(model_name="WideResNet", input_shape = dataset.input_shape, 
                num_classes = dataset.num_classes, sub_policies=sub_policies, model_size= 28, dropout_rate=0.1, temperature=0.1)





train_loader = torch.utils.data.DataLoader(
    dataset.datasets['train'],
    batch_size=64,
    pin_memory=True,
    num_workers=2,
)

validation_loader = torch.utils.data.DataLoader(
    dataset.datasets['val'],
    batch_size=64,
    pin_memory=True,
    num_workers=2,
)

training_config = Config(
    type="darts",
    retain_graph=True,
    log_step=100,
    unroll_steps=1,
    allow_unused=True
)

optimization_config = Config(
    type="darts",
    retain_graph=True,
    log_step=100,
    allow_unused=True,
    unroll_steps=1,
)

tuning_config = Config(
    type="darts",
    retain_graph=True,
    unroll_steps=1,
    log_step=100,
    allow_unused=True,
)


network_training = NetworkTraining(
    name="network_training",
    module=model,
    train_data_loader=train_loader,
    config=training_config,
)

hpo = HyperparameterOptimization(
    name="hpo",
    module=hyperparameters,
    optimizer=hyper_optimizer,
    train_data_loader=validation_loader,
    config=optimization_config,
)

datatuning = DataTuning(
    name="datatuning",
    module=data_net,
    optimizer=data_optimizer,
    train_data_loader=validation_loader,
    config=tuning_config,
)


class HPOEngine(Engine):
    def __init__(self, problems, config=None, dependencies=None, env=None):
        super().__init__(problems, config=config, dependencies=dependencies, env=env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def validation(self):
        corrects = 0
        total = 0
        for x, target in validation_loader:
            x, target = x.to(self.device), target.to(self.device, non_blocking=True)
            alphas = self.arch()
            _, correct = self.network_training.loss(x, alphas, target, acc=True)
            corrects += correct
            total += x.size(0)
        acc = corrects / total

        alphas = self.arch()
        torch.save({"genotype": self.network_training.genotype(alphas)}, "genotype.t7")
        return {"acc": acc}


engine_config = EngineConfig(train_iters=10000, logger_type="none")


problems = [datatuning, network_training]
u2l = {datatuning: [network_training]}
l2u = {network_training: [datatuning]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = HPOEngine(config=engine_config, problems=problems, dependencies=dependencies)

engine.run()

# Close wandb run
wandb.finish()