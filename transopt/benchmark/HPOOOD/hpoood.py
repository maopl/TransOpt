# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import random

from torchvision import datasets, transforms

from typing import Dict, Union


from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.optimizer.sampler.random import RandomSampler
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.benchmark.HPOOOD.hparams_registry import random_hparams, default_hparams, get_hparams



class HPOOOD_base(NonTabularProblem):
    DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # Spawrious datasets
    "SpawriousO2O_easy",
    "SpawriousO2O_medium",
    "SpawriousO2O_hard",
    "SpawriousM2M_easy",
    "SpawriousM2M_medium",
    "SpawriousM2M_hard",
    ]

    problem_type = 'hpoood'
    num_variables = 10
    num_objectives = 1
    workloads = []
    fidelity = None
    
    ALGORITHMS = [
        'ERM',
        'Fish',
        'IRM',
        'GroupDRO',
        'Mixup',
        'MLDG',
        'CORAL',
        'MMD',
        'DANN',
        'CDANN',
        'MTL',
        'SagNet',
        'ARM',
        'VREx',
        'RSC',
        'SD',
        'ANDMask',
        'SANDMask',
        'IGA',
        'SelfReg',
        "Fishr",
        'TRM',
        'IB_ERM',
        'IB_IRM',
        'CAD',
        'CondCAD',
        'Transfer',
        'CausIRL_CORAL',
        'CausIRL_MMD',
        'EQRM',
        'RDM',
        'ADRMX',
    ]

    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):
        self.dataset = HPOOOD_base.DATASETS[workload]
        self.algorithm = kwargs['algorithm']
        self.test_envs = [0]
        super(HPOOOD_base, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
            
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        hparams = default_hparams(self.algorithm, self.dataset)
        
        if self.dataset in vars(datasets):
            dataset = vars(datasets)[self.dataset](self.data_dir,
                self.test_envs, hparams)
        else:
            raise NotImplementedError
        
        in_splits = []
        out_splits = []
        uda_splits = []
        
        # for env_i, env in enumerate(dataset):
        #     uda = []

        #     out, in_ = misc.split_dataset(env,
        #         int(len(env)*args.holdout_fraction),
        #         misc.seed_hash(args.trial_seed, env_i))

        #     if env_i in args.test_envs:
        #         uda, in_ = misc.split_dataset(in_,
        #             int(len(in_)*args.uda_holdout_fraction),
        #             misc.seed_hash(args.trial_seed, env_i))



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
        variables=[Continuous('lr', [-8.0, 0.0]),
            Continuous('weight_decay', [-10.0, -5.0]),
            ]
        ss = SearchSpace(variables)
        self.hparam = ss
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
    def get_score(self, configuration: dict):
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        hparams = default_hparams(self.algorithm, self.dataset)
        hparams['lr'] = configuration["lr"]
        hparams['weight_decay'] = configuration["weight_decay"]
        
    
            


        
        os.makedirs(args.output_dir, exist_ok=True)
            
        if self.dataset in vars(datasets):
            dataset = vars(datasets)[self.dataset](self.data_dir, args.test_envs, hparams)
        else:
            raise NotImplementedError
        pass
    
        in_splits = []
        out_splits = []
        uda_splits = []
        for env_i, env in enumerate(dataset):
            uda = []

            out, in_ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

            if env_i in args.test_envs:
                uda, in_ = misc.split_dataset(in_,
                    int(len(in_)*args.uda_holdout_fraction),
                    misc.seed_hash(args.trial_seed, env_i))

            if hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(in_)
                out_weights = misc.make_weights_for_balanced_classes(out)
                if uda is not None:
                    uda_weights = misc.make_weights_for_balanced_classes(uda)
            else:
                in_weights, out_weights, uda_weights = None, None, None
            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))
            if len(uda):
                uda_splits.append((uda, uda_weights))
    
    
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
    

@problem_registry.register("ERMOOD")
class ERMOOD(HPOOOD_base):
    pass



if __name__ == "__main__":
    p = ERMOOD(task_name='', budget_type='FEs', budget=100, seed = 0, workload = 2, algorithm='ERM')
    

