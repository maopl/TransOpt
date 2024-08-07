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
import collections
import time
import json
import shutil
import hashlib
import copy


from torchvision import datasets, transforms

from typing import Dict, Union


from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.optimizer.sampler.random import RandomSampler
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.benchmark.HPOOOD.hparams_registry import random_hparams, default_hparams, get_hparams
from transopt.benchmark.HPOOOD import datasets
from transopt.benchmark.HPOOOD import misc
from transopt.benchmark.HPOOOD import algorithms
from transopt.benchmark.HPOOOD.fast_data_loader import InfiniteDataLoader, FastDataLoader


def make_record(step, hparams_seed, envs):
    """envs is a list of (in_acc, out_acc, is_test_env) tuples"""
    result = {
        'args': {'test_envs': [], 'hparams_seed': hparams_seed},
        'step': step
    }
    for i, (in_acc, out_acc, is_test_env) in enumerate(envs):
        if is_test_env:
            result['args']['test_envs'].append(i)
        result[f'env{i}_in_acc'] = in_acc
        result[f'env{i}_out_acc'] = out_acc
    return result


class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', '-m', 'domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')
        
        

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
        self.dataset_name = HPOOOD_base.DATASETS[workload]
        self.algorithm_name = kwargs['algorithm']
        self.test_envs = [0]
        self.data_dir = '~/transopt_files/data/'
        self.output_dir = '~/transopt_files/output/'
        self.holdout_fraction = 0.2
        self.uda_holdout_fraction = 0
        self.task = 'domain_generalization'
        self.steps = 500
        self.checkpoint_freq = 50
        
        self.save_model_every_checkpoint = True
        
        self.skip_model_save = False
        
        self.trial_seed = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
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
        
        self.hparams = default_hparams(self.algorithm_name, self.dataset_name)
        
        if self.dataset_name in vars(datasets):
            self.dataset = vars(datasets)[self.dataset_name](self.data_dir,
                self.test_envs, self.hparams)
        else:
            raise NotImplementedError
        
        in_splits = []
        out_splits = []
        uda_splits = []
        
        for env_i, env in enumerate(self.dataset):
            uda = []

            out, in_ = misc.split_dataset(env,
                int(len(env)*self.holdout_fraction),
                misc.seed_hash(self.seed, env_i))

            if env_i in self.test_envs:
                uda, in_ = misc.split_dataset(in_,
                    int(len(in_)*self.uda_holdout_fraction),
                    misc.seed_hash(self.trial_seed, env_i))

            if self.hparams['class_balanced']:
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
            if self.task == "domain_adaptation" and len(uda_splits) == 0:
                raise ValueError("Not enough unlabeled samples for domain adaptation.")

        self.train_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(in_splits)
            if i not in self.test_envs]

        self.uda_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(uda_splits)]

        self.eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=self.dataset.N_WORKERS)
            for env, _ in (in_splits + out_splits + uda_splits)]
         
        self.eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
        self.eval_loader_names = ['env{}_in'.format(i)
            for i in range(len(in_splits))]
        self.eval_loader_names += ['env{}_out'.format(i)
            for i in range(len(out_splits))]
        self.eval_loader_names += ['env{}_uda'.format(i)
            for i in range(len(uda_splits))]
        
        algorithm_class = algorithms.get_algorithm_class(self.algorithm_name)
        self.algorithm = algorithm_class(self.dataset.input_shape, self.dataset.num_classes,
            len(self.dataset) - len(self.test_envs), self.hparams)
        
        self.train_minibatches_iterator = zip(*self.train_loaders)
        self.uda_minibatches_iterator = zip(*self.uda_loaders)
        self.checkpoint_vals = collections.defaultdict(lambda: [])

        self.steps_per_epoch = min([len(env)/self.hparams['batch_size'] for env,_ in in_splits])
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.algorithm.to(self.device)



    def save_checkpoint(self, filename):
        if self.skip_model_save:
            return
        save_dict = {
            "model_input_shape": self.dataset.input_shape,
            "model_num_classes": self.dataset.num_classes,
            "model_num_domains": len(self.dataset) - len(self.test_envs),
            "model_hparams": self.hparams,
            "model_dict": self.algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(self.output_dir, filename))

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
    def train(self, configuration: dict):
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.hparams = default_hparams(self.algorithm_name, self.dataset_name)
        self.hparams['lr'] = configuration["lr"]
        self.hparams['weight_decay'] = configuration["weight_decay"]

        n_steps = self.steps or self.dataset.N_STEPS
        
        last_results_keys = None
        
        start_step = 0
        
        for step in range(start_step, n_steps):
            step_start_time = time.time()
            minibatches_device = [(x.to(self.device), y.to(self.device))
                for x,y in next(self.train_minibatches_iterator)]
            if self.task == "domain_adaptation":
                uda_device = [x.to(self.device)
                    for x,_ in next(self.uda_minibatches_iterator)]
            else:
                uda_device = None
            step_vals = self.algorithm.update(minibatches_device, uda_device)
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

                evals = zip(self.eval_loader_names, self.eval_loaders, self.eval_weights)
                for name, loader, weights in evals:
                    acc = misc.accuracy(self.algorithm, loader, weights, self.device)
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

                epochs_path = os.path.join(self.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                start_step = step + 1
                
                if self.save_model_every_checkpoint:
                    self.save_checkpoint(f'model_step{step}.pkl')
        
        self.save_checkpoint('model.pkl')
        with open(os.path.join(self.output_dir, 'done'), 'w') as f:
            f.write('done')
            
    def get_score(self, configuration: dict):
        self.train(configuration)
        



        


    
    
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
    configuration = {
        "lr": -0.3,
        "weight_decay": -5,
    }
    p.f(configuration=configuration)
    
