import collections
import os
import random
import time
import json
from typing import Dict, Union

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
from transopt.optimizer.sampler.random import RandomSampler
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.benchmark.HPO import algorithms
from transopt.benchmark.HPO.misc import LossPlotter
from transopt.benchmark.HPO.hparams_registry import get_hparams
from transopt.benchmark.HPO.datasets import RobCifar10, RobCifar100, RobImageNet
from transopt.benchmark.HPO import hyperparameter_register





class HPO_base(NonTabularProblem):
    problem_type = 'hpo'
    num_variables = 3
    num_objectives = 1
    workloads = []
    fidelity = None
    
    ALGORITHMS = [
        'ERM',
    ]
    
    DATASETS = [
    "RobCifar10",
    "RobCifar100",
    "RobImageNet",
    ]

    def __init__(
        self, task_name, budget_type, budget, seed, workload, algorithm, network_type='densenet'
        ):
        self.dataset_name = HPO_base.DATASETS[workload]
        self.algorithm_name = algorithm
        
        self.validate_fraction = 0.1
        self.task = 'domain_generalization'
        self.steps = 1000
        self.checkpoint_freq = 50
        self.query = 0
        self.network_type = network_type
        self.hparams = {}
        
        self.save_model_every_checkpoint = False
        
        self.skip_model_save = False
        
        self.trial_seed = seed
        
        user_home = os.path.expanduser('~')
        self.model_save_dir  = os.path.join(user_home, f'transopt_tmp/output/models/{self.algorithm_name}_{self.dataset_name}_{seed}/')
        self.results_save_dir  = os.path.join(user_home, f'transopt_tmp/output/results/{self.algorithm_name}_{self.dataset_name}_{seed}/')
        
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

        self.hparams['batch_size'] = 64
        self.hparams['nonlinear_classifier'] = False

        if self.dataset_name in globals():
            self.dataset = globals()[self.dataset_name](root=None, augment=True)
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented")

        # Remove the splitting of the dataset
        self.train_loaders = [InfiniteDataLoader(
            dataset=self.dataset.datasets,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)]

        self.eval_loaders = [FastDataLoader(
            dataset=self.dataset.get_test_set('standard'),
            batch_size=64,
            num_workers=self.dataset.N_WORKERS)]

        self.eval_loader_names = ['standard']

        # Add corruption test loaders
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        for corruption in corruptions:
            corruption_dataset = self.dataset.get_test_set(f'corruption_{corruption}')
            if corruption_dataset is not None:
                self.eval_loaders.append(FastDataLoader(
                    dataset=corruption_dataset,
                    batch_size=64,
                    num_workers=self.dataset.N_WORKERS))
                self.eval_loader_names.append(f'corruption_{corruption}')

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

        hparams = get_hparams(self.algorithm_name, self.dataset_name)
        variables = []

        for name, (default, min_val, max_val) in hparams.items():
            if min_val is not None and max_val is not None:
                if isinstance(default, int):
                    variables.append(Integer(name, [min_val, max_val]))
                elif isinstance(default, float):
                    if name in ['lr', 'weight_decay']:
                        # Use log scale for learning rate and weight decay
                        variables.append(Continuous(name, [np.log10(min_val), np.log10(max_val)]))
                    else:
                        variables.append(Continuous(name, [min_val, max_val]))
                elif isinstance(default, bool):
                    variables.append(Categorical(name, [True, False]))

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
        print(self.steps)

        n_steps = self.steps or self.dataset.N_STEPS
        
        last_results_keys = None
    
        # loss_plotter = LossPlotter()

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
                        correct = 0
                        total = 0
                        weights_offset = 0

                        self.algorithm.eval()
                        with torch.no_grad():
                            for x, y in loader:
                                x = x.to(self.device)
                                y = y.to(self.device)
                                p = self.algorithm.predict(x)
                                if p.size(1) == 1:
                                    correct += (p.gt(0).eq(y).float()).sum().item()
                                else:
                                    correct += (p.argmax(1).eq(y).float()).sum().item()
                                total += torch.ones(len(x)).sum().item()
                        self.algorithm.train()
                        
                        results[name+'_acc'] = correct / total

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
                    
                    loss_plotter.show()


                if self.save_model_every_checkpoint:
                    self.save_checkpoint(f'model_step{step}.pkl')
        

        
        self.save_checkpoint('model.pkl')
        with open(os.path.join(self.model_save_dir, 'done'), 'w') as f:
            f.write('done')

        return results

    def get_score(self, configuration: dict):
        for key, value in configuration.items():
            self.hparams[key] = value

        algorithm_class = algorithms.get_algorithm_class(self.algorithm_name)
        self.algorithm = algorithm_class(self.dataset.input_shape, self.dataset.num_classes, self.network_type, self.hparams)
        self.algorithm.to(self.device)
        
        self.query += 1
        results = self.train(configuration)
        
        epochs_path = os.path.join(self.results_save_dir, f"{self.query}_lr_{configuration['lr']}_weight_decay_{configuration['weight_decay']}.jsonl")
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")


        val_acc = results['standard_acc']
        test_acc = np.mean([results[f'corruption_{c}_acc'] for c in [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]])
        
        return val_acc, test_acc
        

    def objective_function(
        self,
        configuration,
        fidelity = None,
        seed = None,
        **kwargs
    ) -> Dict:

        if fidelity is None:
            fidelity = {"epoch": 30}
        
        # Convert log scale values back to normal scale
        c = {}
        for key, value in configuration.items():
            if key in ['lr', 'weight_decay']:
                c[key] = 10 ** value
            else:
                c[key] = value
        
        # Add fidelity (epoch) to the configuration
        c["epoch"] = fidelity["epoch"]
        
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
        super(HPO_ERM, self).__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed=seed, workload=workload, algorithm='ERM')

@problem_registry.register("HPO_ROBERM")
class HPO_ROBERM(HPO_base):    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):
        super(HPO_ROBERM, self).__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed=seed, workload=workload, algorithm='ERM')


def test_hpo_roberm():
    print("Testing HPO_ROBERM...")
    try:
        # Create an instance of HPO_ROBERM
        hpo = HPO_ROBERM(task_name='test_roberm', budget_type='FEs', budget=100, seed=0, workload=0)
        
        # Get the configuration space
        config_space = hpo.get_configuration_space()
        
        # Get the fidelity space
        fidelity_space = hpo.get_fidelity_space()
        
        # Manually sample a random configuration
        config = {}
        for name, var in config_space.get_design_variables().items():
            if isinstance(var, Integer):
                config[name] = np.random.randint(var.search_space_range[0], var.search_space_range[1] + 1)
            elif isinstance(var, Continuous):
                config[name] = np.random.uniform(var.search_space_range[0], var.search_space_range[1])
            elif isinstance(var, Categorical):
                config[name] = np.random.choice(var.search_space_range)

        # Manually sample a random fidelity
        fidelity = {}
        for name, var in fidelity_space.get_fidelity_range().items():
            if isinstance(var, Integer):
                fidelity[name] = np.random.randint(var.search_space_range[0], var.search_space_range[1] + 1)
            elif isinstance(var, Continuous):
                fidelity[name] = np.random.uniform(var.search_space_range[0], var.search_space_range[1])
            elif isinstance(var, Categorical):
                fidelity[name] = np.random.choice(var.search_space_range)
        
        # Run the objective function
        result = hpo.objective_function(configuration=config, fidelity=fidelity)
        
        print(f"Configuration: {config.get_dictionary()}")
        print(f"Fidelity: {fidelity.get_dictionary()}")
        print(f"Result: {result}")
        
        assert 'function_value' in result, "Result should contain 'function_value'"
        assert 0 <= result['function_value'] <= 1, "Function value should be between 0 and 1"
        
        print("HPO_ROBERM test passed successfully!")
    except Exception as e:
        print(f"Error occurred during HPO_ROBERM test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Run the test
    test_hpo_roberm()

