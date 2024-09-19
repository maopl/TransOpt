import collections
import os
import random
import time
import json
from typing import Dict, Union
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from transopt.benchmark.HPO import datasets

import transopt.benchmark.HPO.misc as misc
from transopt.agent.registry import problem_registry
from transopt.benchmark.HPO.fast_data_loader import (FastDataLoader,
                                                     InfiniteDataLoader)
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.benchmark.HPO import algorithms
from transopt.benchmark.HPO.hparams_registry import get_hparam_space
from transopt.benchmark.HPO.networks import SUPPORTED_ARCHITECTURES

class HPO_base(NonTabularProblem):
    problem_type = 'hpo'
    num_variables = 3
    num_objectives = 1
    workloads = []
    fidelity = None
    
    ALGORITHMS = [
        'ERM',
        # 'BayesianNN',
        # 'GLMNet'
    ]
    
    ARCHITECTURES = SUPPORTED_ARCHITECTURES
    
    DATASETS = [
    "RobCifar10",
    # "RobCifar100",
    # "RobImageNet",
    ]

    def __init__(
        self, task_name, budget_type, budget, seed, workload, algorithm, architecture, model_size
        ):
        
        # Check if algorithm is valid
        if algorithm not in HPO_base.ALGORITHMS:
            raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {HPO_base.ALGORITHMS}")
        self.algorithm_name = algorithm

        # Check if workload is valid
        if workload < 0 or workload >= len(HPO_base.DATASETS):
            raise ValueError(f"Invalid workload: {workload}. Must be between 0 and {len(HPO_base.DATASETS) - 1}")
        self.dataset_name = HPO_base.DATASETS[workload]

        # Check if architecture is valid
        if architecture not in HPO_base.ARCHITECTURES:
            raise ValueError(f"Invalid architecture: {architecture}. Must be one of {list(HPO_base.ARCHITECTURES.keys())}")
        if model_size not in HPO_base.ARCHITECTURES[architecture]:
            raise ValueError(f"Invalid model_size: {model_size} for architecture: {architecture}. Must be one of {HPO_base.ARCHITECTURES[architecture]}")
        self.architecture = architecture
        self.model_size = model_size

        super(HPO_base, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
        
        self.query = 0
        self.trial_seed = seed
        self.hparams = {}
        
        user_home = os.path.expanduser('~')
        self.model_save_dir  = os.path.join(user_home, f'transopt_tmp/output/models/{self.algorithm_name}_{self.architecture}_{self.model_size}_{self.dataset_name}_{seed}/')
        self.results_save_dir  = os.path.join(user_home, f'transopt_tmp/output/results/{self.algorithm_name}_{self.architecture}_{self.model_size}_{self.dataset_name}_{seed}/')
        
        print(f"Selected algorithm: {self.algorithm_name}, dataset: {self.dataset_name}")
        print(f"Model architecture: {self.architecture}")
        if hasattr(self, 'model_size'):
            print(f"Model size: {self.model_size}")
        else:
            print("Model size not specified")
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.dataset_name in vars(datasets):
            self.dataset = vars(datasets)[self.dataset_name](root=None, augment=True)
        else:
            raise NotImplementedError

        # if self.dataset_name in globals():
        #     self.dataset = globals()[self.dataset_name](root=None, augment=True)
        # else:
        #     raise NotImplementedError(f"Dataset {self.dataset_name} not implemented")

        
        self.checkpoint_vals = collections.defaultdict(lambda: [])
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.trial_seed}")

        else:
            self.device = "cpu"
    def create_data_loaders(self, batch_size):
        train_loaders = FastDataLoader(
            dataset=self.dataset.datasets['train'],
            batch_size=batch_size,
            num_workers=2)  # Assuming N_WORKERS is 2, adjust if needed
        
        val_loaders = FastDataLoader(
            dataset=self.dataset.datasets['val'],
            batch_size=batch_size,
            num_workers=2)  # Assuming N_WORKERS is 2, adjust if needed

        eval_loaders = []
        eval_loader_names = []

        # Get all available test set names
        available_test_sets = self.dataset.get_available_test_set_names()

        for test_set_name in available_test_sets:
            if test_set_name.startswith('test_'):
                eval_loaders.append(FastDataLoader(
                    dataset=self.dataset.datasets[test_set_name],
                    batch_size=batch_size,
                    num_workers=2))  # Assuming N_WORKERS is 2, adjust if needed
                eval_loader_names.append(test_set_name)

        return train_loaders, val_loaders, eval_loaders, eval_loader_names

    def save_checkpoint(self, filename):
        save_dict = {
            "model_input_shape": self.dataset.input_shape,
            "model_num_classes": self.dataset.num_classes,
            "model_hparams": self.hparams,
            "model_dict": self.algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(self.model_save_dir, filename))
        
    def get_configuration_space(
        self, seed: Union[int, None] = None):

        hparam_space = get_hparam_space(self.algorithm_name)
        variables = []

        for name, (hparam_type, range) in hparam_space.items():
            if hparam_type == 'categorical':
                variables.append(Categorical(name, range))
            elif hparam_type == 'float':
                variables.append(Continuous(name, range))
            elif hparam_type == 'int':
                variables.append(Integer(name, range))
            elif hparam_type == 'log':
                variables.append(LogContinuous(name, range))

        ss = SearchSpace(variables)
        return ss
    
    def get_fidelity_space(
        self, seed: Union[int, None] = None):

        fs = FidelitySpace([
            Integer("epoch", [1, 100])  # Adjust the range as needed
        ])
        return fs
    
    def train(self, configuration: dict):
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.epoches = configuration['epoch']
        print(f"Total epochs: {self.epoches}")
        
        last_results_keys = None
        
        self.train_loader, self.val_loader, self.eval_loaders, self.eval_loader_names = self.create_data_loaders(self.hparams['batch_size'])
        self.hparams['nonlinear_classifier'] = False
    
        for epoch in range(self.epoches):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            self.algorithm.train()
            total_batches = len(self.train_loader)
            for x, y in tqdm(self.train_loader, total=total_batches, desc=f"Epoch {epoch+1}/{self.epoches}", unit="batch"):
                step_start_time = time.time()
                minibatches_device = [(x.to(self.device), y.to(self.device))]

                step_vals = self.algorithm.update(minibatches_device)
                self.checkpoint_vals['step_time'].append(time.time() - step_start_time)

                for key, val in step_vals.items():
                    self.checkpoint_vals[key].append(val)
                
                # Update epoch statistics
                epoch_loss += step_vals.get('loss', 0.0)
                epoch_correct += step_vals.get('correct', 0)
                epoch_total += sum(len(x) for x, _ in minibatches_device)

            # Compute epoch metrics
            epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            epoch_loss /= len(self.train_loader)
            epoch_time = time.time() - epoch_start_time

            # Evaluate on validation set
            val_acc = self.evaluate_loader(self.val_loader)

            # Prepare epoch results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc,
                'val_acc': val_acc,
                'epoch_time': epoch_time,
            }

            # Evaluate on all test loaders
            for name, loader in zip(self.eval_loader_names, self.eval_loaders):
                epoch_results[f'{name}_acc'] = self.evaluate_loader(loader)
                

            # Calculate memory usage
            epoch_results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.**3)

            # Print epoch results
            results_keys = sorted(epoch_results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([epoch_results[key] for key in results_keys], colwidth=12)

            # Update results with hyperparameters
            epoch_results['hparams'] = self.hparams
            # Save epoch results
            self.save_epoch_results(epoch_results)
            

        # Save final checkpoint and mark as done
        self.save_checkpoint('model.pkl')
        with open(os.path.join(self.model_save_dir, 'done'), 'w') as f:
            f.write('done')

        return epoch_results

    def save_epoch_results(self, results):
        epoch_path = os.path.join(self.results_save_dir, f"epoch_{results['epoch']}.json")
        with open(epoch_path, 'w') as f:
            json.dump(results, f, indent=2)

    def evaluate_loader(self, loader):
        self.algorithm.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                p = self.algorithm.predict(x)
                correct += (p.argmax(1).eq(y) if p.size(1) != 1 else p.gt(0).eq(y)).float().sum().item()
                total += len(x)
        self.algorithm.train()
        return correct / total

    def get_score(self, configuration: dict):
        for key, value in configuration.items():
            self.hparams[key] = value

        algorithm_class = algorithms.get_algorithm_class(self.algorithm_name)
        self.algorithm = algorithm_class(self.dataset.input_shape, self.dataset.num_classes, self.architecture, self.model_size, self.hparams)
        self.algorithm.to(self.device)
        
        self.query += 1
        results = self.train(configuration)
        
        epochs_path = os.path.join(self.results_save_dir, f"{self.query}_lr_{configuration['lr']}_weight_decay_{configuration['weight_decay']}.jsonl")
        with open(epochs_path, 'w') as f:
            json.dump(results, f, indent=2)

        val_acc = results['val_acc']
        
        return val_acc
        

    def objective_function(
        self,
        configuration,
        fidelity = None,
        seed = None,
        **kwargs
    ) -> Dict:

        if fidelity is None:
            fidelity = {"epoch": 50}
        
        # Convert log scale values back to normal scale
        c = self.configuration_space.map_to_design_space(configuration)
        
        # Add fidelity (epoch) to the configuration
        c["epoch"] = fidelity["epoch"]
        c['batch_size'] = 64
        
        c['data_augmentation'] = True
        c['class_balanced'] = True
        c['nonlinear_classifier'] = False
        
        val_acc = self.get_score(c)

        results = {list(self.objective_info.keys())[0]: float(val_acc)}
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
        algorithm = kwargs.get('algorithm', 'ERM')
        architecture = kwargs.get('architecture', 'resnet')
        model_size = kwargs.get('model_size', 18)
        
        super(HPO_ERM, self).__init__(
            task_name=task_name, 
            budget_type=budget_type, 
            budget=budget, 
            seed=seed, 
            workload=workload, 
            algorithm=algorithm, 
            architecture=architecture, 
            model_size=model_size
        )

def test_all_combinations():
    print("Testing all combinations of architectures, algorithms, and datasets...")
    
    for architecture in HPO_base.ARCHITECTURES:
        for model_size in HPO_base.ARCHITECTURES[architecture]:
            for algorithm in HPO_base.ALGORITHMS:
                for dataset_index, dataset in enumerate(HPO_base.DATASETS):
                    print(f"Testing {architecture}-{model_size} with {algorithm} on {dataset}...")
                    try:
                        # Create an instance of HPO_base
                        hpo = HPO_base(task_name='test_combination', 
                                       budget_type='FEs', budget=100, seed=0, 
                                       workload=dataset_index, algorithm=algorithm, 
                                       architecture=architecture, model_size=model_size)
                        
                        # Get the configuration space
                        config_space = hpo.get_configuration_space()
                        
                        # Get the fidelity space
                        fidelity_space = hpo.get_fidelity_space()
                        
                        # Sample a random configuration
                        config = {}
                        for name, var in config_space.get_design_variables().items():
                            if isinstance(var, Integer):
                                config[name] = np.random.randint(var.search_space_range[0], var.search_space_range[1] + 1)
                            elif isinstance(var, Continuous) or isinstance(var, LogContinuous):
                                config[name] = np.random.uniform(var.search_space_range[0], var.search_space_range[1])
                            elif isinstance(var, Categorical):
                                config[name] = np.random.choice(var.search_space_range)
                                
                        
                        
                        # Sample a random fidelity
                        fidelity = {}
                        for name, var in fidelity_space.get_fidelity_range().items():
                            if isinstance(var, Integer):
                                fidelity[name] = np.random.randint(var.search_space_range[0], var.search_space_range[1] + 1)
                            elif isinstance(var, Continuous):
                                fidelity[name] = np.random.uniform(var.search_space_range[0], var.search_space_range[1])
                            elif isinstance(var, Categorical):
                                fidelity[name] = np.random.choice(var.search_space_range)
                        
                        # Set a small epoch for quick testing
                        fidelity['epoch'] = 2
                        
                        # Run the objective function
                        result = hpo.objective_function(configuration=config, fidelity=fidelity)
                        
                        print(f"Configuration: {config}")
                        print(f"Fidelity: {fidelity}")
                        print(f"Result: {result}")
                        
                        assert list(hpo.get_objectives().keys())[0] in result, f"Result should contain '{list(hpo.get_objectives().keys())[0]}'"
                        assert 0 <= result[list(hpo.get_objectives().keys())[0]] <= 1, f"{list(hpo.get_objectives().keys())[0]} should be between 0 and 1"
                        
                        print(f"Test passed for {architecture}-{model_size} with {algorithm} on {dataset}!")
                        print("--------------------")
                    except Exception as e:
                        print(f"Error occurred during test for {architecture}-{model_size} with {algorithm} on {dataset}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        print("--------------------")

if __name__ == "__main__":
    import torch
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Run the comprehensive test
    try:
        test_all_combinations()
    except Exception as e:
        print(f"Error occurred during HPO_ERM test: {str(e)}")
        import traceback
        traceback.print_exc()



