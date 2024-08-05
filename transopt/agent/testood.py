import logging
import time
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm

from torchvision import datasets, transforms

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.optimizer.sampler.random import RandomSampler
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.utils.openml_data_manager import OpenMLHoldoutDataManager
from transopt.datamanager.database import Database
from services import Services

from transopt.benchmark.HPO.HPOCNN import *

import os
import sys
import unittest
from pathlib import Path


current_dir = Path(__file__).resolve().parent
package_dir = current_dir.parent
sys.path.insert(0, str(package_dir))

def setUp():
    db = Database("database.db")
    table_name = "test_table"
        
        

if __name__ == "__main__":

    services = Services(None, None, None)
    task_name = []
    parameters = []
    tables = services.get_experiment_datasets()
    for table in tables:
        print(table[1]['data_number'])
        if table[1]['data_number'] == 100:
            task_name = table[0]
            print(task_name)

            all_data = services.data_manager.db.select_data(task_name)
            table_info = services.data_manager.db.query_dataset_info(task_name)
                    
            objectives = table_info["objectives"]
            ranges = [tuple(var['range']) for var in table_info["variables"]]
            initial_number = table_info["additional_config"]["initial_number"]
            obj = objectives[0]["name"]
            obj_type = objectives[0]["type"]

            obj_data = [data[obj] for data in all_data]
            max_id = np.argmax(obj_data)
            
            var_data = [[data[var["name"]] for var in table_info["variables"]] for data in all_data]
            variables = [var["name"] for var in table_info["variables"]]
            ret = {}
            traj = services.construct_trajectory_data(task_name, obj_data, obj_type="maximize")
            best_var = var_data[max_id]
            lr = np.exp2(best_var[0])
            momentum = best_var[1]
            weight_decay = np.exp2(best_var[2])
            parameters.append((lr, momentum, weight_decay))
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(1)
    else:
        device = torch.device("cpu")
        
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
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    )
    
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False
    )
    epochs = 30
    batch_size = 64

    # lr = 0.0017607943222948076
    # momentum = 0.6997583600209312
    # weight_decay = 0.004643925899318933
    
    lr = parameters[0][0]
    momentum = parameters[0][1]
    weight_decay = parameters[0][2]
    print(lr, momentum, weight_decay)

    net = Learner(target_classes=10).to(device)
    net.load_state_dict(torch.load(f'./temp_model/CNN_101/lr_0.9022972086386453_momentum_0.9987180709134317_weight_decay_0.006041134854023943_model.pth'))
    # criterion = nn.NLLLoss()
    # optimizer = optim.SGD(
    #     net.parameters(),
    #     lr=lr,
    #     momentum=momentum,
    #     weight_decay = weight_decay,
    # )
    # start_time = time.time()
    # for e in tqdm.tqdm(range(epochs)):
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         optimizer.zero_grad()

    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #     print("Epoch %d, Loss: %.3f" % (e + 1, running_loss / len(trainloader)))

    correct = 0
    total = 0
    import os

    # os.makedirs(f'./temp_model/test', exist_ok=True)

    # model_save_path = f'./temp_model/test/lr_{lr}_momentum_{momentum}_weight_decay_{weight_decay}_model.pth'  # 自定义保存路径
    # torch.save(net.state_dict(), model_save_path)



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


            

            
        

    # model_save_path = './temp_model/model.pth'
    # torch.save(net.state_dict(), model_save_path)