import collections
import os
import random
import time
import json
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from transopt.benchmark.HPO import datasets

import transopt.benchmark.HPO.misc as misc
from transopt.agent.registry import problem_registry
from transopt.benchmark.HPO.fast_data_loader import (FastDataLoader,
                                                     InfiniteDataLoader)
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.optimizer.sampler.random import RandomSampler
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.benchmark.HPO import algorithms


from transopt.benchmark.HPO import hyperparameter_register





class HPO_base(NonTabularProblem):
    problem_type = 'hpoood'
    num_variables = 10
    num_objectives = 1
    workloads = []
    fidelity = None
    
    
    ALGORITHMS = [
        'ERM',
        
    ]
    
    DATASETS = [
    "RobMNIST",
    "RobCifar10",
    "RobCifar100",
    "RobImageNet",
    ]

    def __init__(
        self, task_name, budget_type, budget, seed, workload, algorithm
        ):
        self.dataset_name = HPO_base.DATASETS[workload]
        self.algorithm_name = algorithm
        self.data_dir = '~/transopt_files/data/'
        self.output_dir = f'~/transopt_files/output/'
        self.validate_fraction = 0.1
        self.task = 'domain_generalization'
        self.steps = 1000
        self.checkpoint_freq = 50
        self.query = 0
        self.hparams = {}
        
        self.save_model_every_checkpoint = False
        
        self.skip_model_save = False
        
        self.trial_seed = seed
        
        self.model_save_dir = self.output_dir + f'models/{self.algorithm_name}_{self.dataset_name}_{seed}/'
        self.results_save_dir = self.output_dir + f'results/{self.algorithm_name}_{self.dataset_name}_{seed}/'
        
        print(f"Selected algorithm: {self.algorithm_name}, dataset: {self.dataset_name}")
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)
        super(HPO_base, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # self.hparams = default_hparams(self.algorithm_name, self.dataset_name)
        
        self.hparams['batch_size'] = 64

        if self.dataset_name in vars(datasets):
            self.dataset = vars(datasets)[self.dataset_name](self.data_dir)
        else:
            raise NotImplementedError
        
        val, in_ = misc.split_dataset(self.dataset,
            int(len(self.dataset)*self.validate_fraction), self.seed)
        
        out = self.dataset.test
        
        out_ds = self.dataset.test_ds

        self.train_loaders = [InfiniteDataLoader(
            dataset=in_,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)]
        
        self.val_loaders = [InfiniteDataLoader(
            dataset=val,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)]

        self.eval_loaders = [FastDataLoader(
            dataset=data,
            batch_size=64,
            num_workers=self.dataset.N_WORKERS)
            for data in (in_, val, out, out_ds)]
    
     
        self.eval_loader_names = ['_in']
        self.eval_loader_names += ['_val']
        self.eval_loader_names += ['_out']
        self.eval_loader_names += ['_out_ds']
        
        
        self.train_minibatches_iterator = zip(*self.train_loaders)
        self.checkpoint_vals = collections.defaultdict(lambda: [])

        self.steps_per_epoch = self.hparams['batch_size']
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.trial_seed}")

        else:
            self.device = "cpu"


    def save_checkpoint(self, filename):
        if self.skip_model_save:
            return
        save_dict = {
            "model_input_shape": self.dataset.input_shape,
            "model_num_classes": self.dataset.num_classes,
            "model_hparams": self.hparams,
            "model_dict": self.algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(self.model_save_dir, filename))
        
    def get_configuration_space(
        self, seed: Union[int, None] = None):

        variables=[Continuous('lr', [-10.0, 0.0]),
            Continuous('weight_decay', [-10.0, -5.0]),
            Continuous('momentum', [-10.0, 0]),
            ]
        ss = SearchSpace(variables)
        self.hparam = ss
        return ss
    
    def get_fidelity_space(
        self, seed: Union[int, None] = None):

        # return fidel_space
        fs = FidelitySpace([])
        return fs
    
    def train(self, configuration: dict):
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.epoches = configuration['epoch']
        print(self.steps)

        n_steps = self.steps or self.dataset.N_STEPS
        
        last_results_keys = None
    
        for epoch in range(self.epoches):
            start_step = 0
            for step in range(start_step, n_steps):
                step_start_time = time.time()
                minibatches_device = [(x.to(self.device), y.to(self.device))
                    for x,y in next(self.train_minibatches_iterator)]

                step_vals = self.algorithm.update(minibatches_device)
                self.checkpoint_vals['step_time'].append(time.time() - step_start_time)

                for key, val in step_vals.items():
                    self.checkpoint_vals[key].append(val)

                if (step % self.checkpoint_freq == 0) or (step == n_steps - 1):
                    results = {
                        'step': step,
                        'epoch': step / self.steps_per_epoch,
                    }

                    for key, val in self.checkpoint_vals.items():
                        results[key] = np.mean(val)

                    evals = zip(self.eval_loader_names, self.eval_loaders)
                    for name, loader in evals:
                        acc = misc.accuracy(self.algorithm, loader, self.device)
                        results[name+'_acc'] = acc

                    results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

                    results_keys = sorted(results.keys())
                    if results_keys != last_results_keys:
                        misc.print_row(results_keys, colwidth=12)
                        last_results_keys = results_keys
                    misc.print_row([results[key] for key in results_keys],
                        colwidth=12)

                    results.update({
                        'hparams': self.hparams,
                    })

                    start_step = step + 1

                if self.save_model_every_checkpoint:
                    self.save_checkpoint(f'model_step{step}.pkl')
        
        self.save_checkpoint('model.pkl')
        with open(os.path.join(self.model_save_dir, 'done'), 'w') as f:
            f.write('done')
        
        return results

    def get_score(self, configuration: dict):
        
        self.hparams['nonlinear_classifier'] = False

        # self.hparams['lr'] = configuration["lr"]
        # self.hparams['weight_decay'] = configuration["weight_decay"]
        
        self.hparams['lr'] = 0.001
        self.hparams['weight_decay'] = 0.00001
        
        algorithm_class = algorithms.get_algorithm_class(self.algorithm_name)
        self.algorithm = algorithm_class(self.dataset.input_shape, self.dataset.num_classes,
            len(self.dataset), self.hparams)
        self.algorithm.to(self.device)
        
        self.query += 1
        results = self.train(configuration)
        
        epochs_path = os.path.join(self.results_save_dir, f"{self.query}_lr_{configuration['lr']}_weight_decay_{configuration['weight_decay']}.jsonl")
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")


        val_acc = [i[1] for i in results.items() if 'val' in i[0]]
        avg_val_acc = np.mean(val_acc)
        
        test_acc = [i[1] for i in results.items() if 'out' in i[0]]
        avg_test_acc = np.mean(test_acc)
        
        return avg_val_acc, avg_test_acc
        

    def objective_function(
        self,
        configuration,
        fidelity = None,
        seed = None,
        **kwargs
    ) -> Dict:

            
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']
        else:
            epoch = 20
            
        if fidelity is None:
            fidelity = {"epoch": epoch, "data_frac": 0.8}
        c = {
            "lr": np.exp(configuration["lr"]),
            "weight_decay": np.exp(configuration["weight_decay"]),
            'momentum': np.exp(configuration["momentum"]),
            "batch_size": 64,
            "epoch": fidelity["epoch"],
        }
        val_acc, test_acc = self.get_score(c)

        results = {list(self.objective_info.keys())[0]: float(1 - val_acc)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results
    
    def get_objectives(self) -> Dict:
        return {'function_value': 'minimize'}
    
    def get_problem_type(self):
        return "hpo"


@problem_registry.register("HPO_ERM")
class HPO_ERM(HPO_base):    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):
        super(HPO_ERM, self).__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed = seed, workload = workload, algorithm='ERM')

if __name__ == "__main__":
    # lr = 0.03326703506193591
    # weight_decay = 0.0015442280272030557
    
    p = HPO_ERM(task_name='', budget_type='FEs', budget=100, seed = 0, workload = 1)
    configuration = {
        "lr": -3,
        "weight_decay": -5,
        "momentum": -3
    }
    p.f(configuration=configuration)
    