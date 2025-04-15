import GPy
import GPyOpt
import numpy as np
import copy
from typing import List, Dict, Union
from transopt.benchmark.HPO.HPO_ERM import DGHPO_ERM
from scipy.stats import norm
from scipy.stats import entropy
from transopt.optimizer.optimizer_base.bo import OptimizerBase
from transopt.optimizer.sampler import RandomSampler
from transopt.optimizer.acquisition_function import AcquisitionEI, AcquisitionConstrainLCB
from transopt.optimizer.model.mtgp import MTGP
from transopt.utils.serialization import (multioutput_to_ndarray,
                                          output_to_ndarray)
from transopt.optimizer.acquisition_function.sequential import Sequential
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *

import torchvision.transforms as transforms
import os
import json

def construct_normalized_space(
    search_space: SearchSpace) -> SearchSpace:

    variables = []

    # Iterate through variables in original search space
    for var_name, var in search_space._variables.items():
        # Create new variable of same type but with [0,1] range
        if var.type == "continuous":
            variables.append(Continuous(var_name, [0.0, 1.0]))
        elif var.type == "integer": 
            variables.append(Integer(var_name, [0, 1]))
        else:
            variables.append(LogContinuous(var_name, [0.0, 1.0]))

    # Construct new normalized search space
    normalized_space = SearchSpace(variables)
    return normalized_space

class BO(OptimizerBase):
    """
    The abstract Model for Bayesian Optimization
    """

    def __init__(self, Sampler, ACF, Model, Normalizer, config, search_space):
        super(BO, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.search_space = search_space
        self.ini_num = 55
        self.Sampler = Sampler
        self.ACF = ACF
        self.Model = Model
        self.Normalizer = None
        self.ACF.link_model(model=self.Model)
        # Create a new search space with ranges normalized to [0,1]
        normalized_space = construct_normalized_space(self.search_space)
        self.ACF.link_space(normalized_space)
        self.evaluator = Sequential(self.ACF, batch_size=1)

    def sample_initial_set(self, search_space, ini_num):
        return self.Sampler.sample(search_space, ini_num)
    
    def fit(self):
        
        self.Model.fit(copy.deepcopy(self._X), copy.deepcopy(self._Y), optimize = True)
    
    
    def suggest(self):
        suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
        # suggested_sample = self.search_space.zip_inputs(suggested_sample)

        if self.Normalizer:
            suggested_sample = self.Normalizer.inverse_transform(X=suggested_sample)[0]
        
        return suggested_sample


    def observe(self, X: np.ndarray, Y: np.ndarray) -> None:
        # Check if the lists are empty and return if they are
        if X.shape[0] == 0 or len(Y) == 0:
            return
        
        self._X = np.vstack((self._X, X)) if self._X.size else X
        self._Y = np.vstack((self._Y, Y)) if self._Y.size else Y
        

def get_initial_data(exp_folder, model_name, model_size, dataset, seed):
    """
    Get initial data from experiment folder for modeling
    Args:
        exp_folder: EXP_4 folder path
        model_name: model architecture name (e.g. alexnet, resnet)
        model_size: model size (e.g. 1, 18)
        dataset: dataset name (e.g. RobCifar10)
        seed: random seed
    Returns:
        X: parameter configurations
        Y: performance metrics
    """
    # Construct subfolder path pattern
    subfolder = f"DGHPO_ERM_ParaAUG_{model_name}_{model_size}_{dataset}_{seed}"
    full_path = os.path.join(exp_folder, subfolder)
    
    if not os.path.exists(full_path):
        raise ValueError(f"Folder {full_path} does not exist")
        
    X = []
    Y = []
    Y_all = []
    # Read all jsonl files in the subfolder
    for file in os.listdir(full_path):
        if file.endswith('.jsonl'):
            with open(os.path.join(full_path, file), 'r') as f:
                data = json.load(f)
                
                # Extract hyperparameters
                hparams = {
                    'lr': np.log10(data['hparams']['lr']),
                    'weight_decay': np.log10(data['hparams']['weight_decay']),
                    'momentum': data['hparams']['momentum'],
                    'batch_size': np.log2(data['hparams']['batch_size']) - 3,
                    'dropout_rate': data['hparams']['dropout_rate'],
                    'mu1': data['hparams']['mu1'],
                    'sigma1': data['hparams']['sigma1'],
                    'mu2': data['hparams']['mu2'], 
                    'sigma2': data['hparams']['sigma2'],
                    'weight': data['hparams']['weight']
                }
                
                # Get all metrics ending with 'acc'
                metrics = []
                for key, value in data.items():
                    if key.endswith('_acc'):
                        metrics.append(value)
                        
                X.append(list(hparams.values()))
                Y.append(data['val_acc'])
                Y_all.append(metrics)
    
    return np.array(X), np.array(Y), np.array(Y_all)

def PE(aug_images):
    # Flatten the image data to fit a Gaussian distribution
    pixel_values = np.array([img[0].flatten().detach().cpu().numpy() for img in aug_images])
    
    # Fit a Gaussian distribution
    mu, sigma = norm.fit(pixel_values)
    
    # Calculate the Shannon entropy of the Gaussian distribution
    ent = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    return ent

def get_data_hparam_space():
    return hpo.get_data_hparam_space()


def get_model_hparam_space():
    return hpo.configuration_space.original_ranges


if __name__ == "__main__":
    seed = 0
    ini_num = 20
    budget = 200
    # Create a single HPO_ERM instance
    hpo = DGHPO_ERM(task_name='DGHPO', budget_type='FEs', budget=budget, seed=42, 
                workload=0, algorithm='ERM_ParaAUG', gpu_id=1, augment=None, architecture='resnet', 
                model_size=18, optimizer='DGHPO', base_dir='/data')

    search_space = hpo.get_search_space()

    param_ranges = []
    for param_name, param_range in search_space.ranges.items():
        if len(param_ranges) < 5:  # Only for paramodel parameters
            param_ranges.append((param_range[0], param_range[1]))

    optimizer = BO(Sampler=RandomSampler(config={}, n_samples=ini_num), ACF=AcquisitionConstrainLCB(config={}), Model=MTGP(config={}), Normalizer=None, config={}, search_space=search_space)


    samples = optimizer.sample_initial_set(search_space=search_space, ini_num=ini_num)

    samples = [search_space.map_to_design_space(sample) for sample in samples]

    observations = []

    for sample in samples:
        observations.append(hpo.objective_function(configuration=sample))
    
    new_Y = np.array([observation['function_value'] for observation in observations])
    new_X = np.array([search_space.map_from_design_space(sample) for sample in samples])

    # samples, observations, observations_all = get_initial_data(exp_folder='./data/EXP_4', model_name='resnet', model_size=18, dataset='RobCifar10', seed=42)
    # new_Y = np.array([observation for observation in observations])
    # new_X = np.array([sample for sample in samples])

    # Split X and normalize paramodel part using search space ranges
    new_X_paramodel = np.array(new_X[:,0:5])
    new_X_aug = np.array(new_X[:,5:])
    
    Y_aug = []
    for i in new_X_aug:
        original_train = hpo.dataset.datasets['train']
        original_val = hpo.dataset.datasets['val']
        
        # Create a new sampler policy with the current policy index
        hpo.augmenter.reset_gaussian(mu1=i[0], sigma1=i[1], mu2=i[2], sigma2=i[3], weight=i[4])
                
        # Apply the selected policy to transform the training data
        transformed_data = []
        for x, y in original_train:
            # Convert tensor to PIL Image
            img = transforms.ToPILImage()(x)
            # Apply sampler transform
            transformed_img = hpo.augmenter(img=img)
            # Convert back to tensor and normalize
            transformed_x = transforms.ToTensor()(transformed_img)
            transformed_x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transformed_x)
            transformed_data.append(transformed_x)
        Y_aug.append(np.array([PE(transformed_data)]))


    # Normalize each column of new_X_paramodel to [0,1]
    for i in range(new_X_paramodel.shape[1]):
        min_val, max_val = param_ranges[i]
        new_X_paramodel[:,i] = (new_X_paramodel[:,i] - min_val) / (max_val - min_val)
        

    Y_paramodel = new_Y
    
    
    Y_aug = np.array(Y_aug)
    Y_aug = (Y_aug - np.mean(Y_aug)) / np.std(Y_aug)

    optimizer.observe(new_X_paramodel, Y_paramodel)

    while (budget):
        optimizer.Model.meta_fit([new_X_aug], [Y_aug])
        optimizer.fit()
        suggested_samples = optimizer.suggest()

        # Denormalize suggested samples for paramodel parameters
        for i in range(suggested_samples.shape[1]):
            if i < 5:  # Only denormalize paramodel parameters
                min_val, max_val = param_ranges[i]
                suggested_samples[:,i] = suggested_samples[:,i] * (max_val - min_val) + min_val

        suggested_samples = [search_space.map_to_design_space(sample) for sample in suggested_samples]
        new_observations = []
        for sample in suggested_samples:
            new_observations.append(hpo.objective_function(configuration=sample))
        new_Y = np.array([observation['function_value'] for observation in new_observations])
        new_X = np.array([search_space.map_from_design_space(sample) for sample in suggested_samples])

        # Split and normalize new X
        new_X_paramodel = np.array(new_X[:,0:5])
        Y_paramodel = new_Y

        optimizer.observe(X=new_X_paramodel, Y=Y_paramodel)
        new_X_aug = np.vstack((new_X_aug, np.array(new_X[:,5:])))
        Y_aug = np.vstack((Y_aug, np.array([PE(hpo.dataset.datasets['train'])]) ))
        Y_aug = (Y_aug - np.mean(Y_aug)) / np.std(Y_aug)
        
        budget -= 1
