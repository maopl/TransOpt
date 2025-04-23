
from transopt.agent.registry import (acf_registry, sampler_registry,
                                     selector_registry, space_refiner_registry,
                                     model_registry, pretrain_registry, normalizer_registry)
from transopt.optimizer.optimizer_base.bo import BO
from transopt.optimizer.optimizer_base.bilevel import Bilevel


def ConstructOptimizer(optimizer_config: dict = None, seed: int = 0) -> BO:
    
    # if 'SpaceRefinerParameters' not in optimizer_config:
    #     optimizer_config['SpaceRefinerParameters'] = {}
    # if 'SamplerParameters' not in optimizer_config:
    #     optimizer_config['SamplerParameters'] = {}
    # if 'ACFParameters' not in optimizer_config:
    #     optimizer_config['ACFParameters'] = {}
    # if 'ModelParameters' not in optimizer_config:
    #     optimizer_config['ModelParameters'] = {}
    # if 'PretrainParameters' not in optimizer_config:
    #     optimizer_config['PretrainParameters'] = {}
    # if 'NormalizerParameters' not in optimizer_config:
    #     optimizer_config['NormalizerParameters'] = {}
    # if 'SamplerInitNum' not in optimizer_config: 
    #     optimizer_config['SamplerInitNum'] = 11
    
    """Create the optimizer object."""
    if optimizer_config['SearchSpace']['type'] == None:
        SpaceRefiner = None
    else:
        if 'Parameters' not in optimizer_config['SearchSpace']:
            optimizer_config['SearchSpace']['Parameters'] = {}
        SpaceRefiner = space_refiner_registry[optimizer_config['SearchSpace']['type']](optimizer_config['SearchSpace']['Parameters'])
    
    if 'Parameters' not in optimizer_config['Initialization']:
        optimizer_config['Initialization']['Parameters'] = {}
    Initialization = sampler_registry[optimizer_config['Initialization']['type']](optimizer_config['Initialization']['InitNum'],optimizer_config['Initialization']['Parameters'])
    
    if 'Parameters' not in optimizer_config['AcquisitionFunction']:
        optimizer_config['AcquisitionFunction']['Parameters'] = {}
    ACF = acf_registry[optimizer_config['AcquisitionFunction']['type']](config = optimizer_config['AcquisitionFunction']['Parameters'])

    if 'Parameters' not in optimizer_config['Model']:
        optimizer_config['Model']['Parameters'] = {}
    Model = model_registry[optimizer_config['Model']['type']](config = optimizer_config['Model']['Parameters'])

    if optimizer_config['Pretrain']['type'] == None:
        Pretrain = None
    else:
        if 'Parameters' not in optimizer_config['Pretrain']:
            optimizer_config['Pretrain']['Parameters'] = {}
        Pretrain = pretrain_registry[optimizer_config['Pretrain']['type']](optimizer_config['Pretrain']['Parameters'])
        

    
    if optimizer_config['Normalizer']['type'] == None:
        Normalizer = None
    else:
        if 'Parameters' not in optimizer_config['Normalizer']:
            optimizer_config['Normalizer']['Parameters'] = {}
        Normalizer = normalizer_registry[optimizer_config['Normalizer']['type']](optimizer_config['Normalizer']['Parameters'])
        
    
    ''' Bugee original code. No 'Optimizer' in optimizer_config
    
    if optimizer_config['Optimizer'] == 'BO':
        optimizer = BO(SpaceRefiner, Sampler, ACF, Pretrain, Model, Normalizer, optimizer_config)
    elif optimizer_config['Optimizer'] == 'Bilevel':
        optimizer = Bilevel(optimizer_config)

    '''
    # Just for test.
    optimizer_type = optimizer_config.get('Optimizer', 'BO')
    if optimizer_type == 'BO':
        optimizer = BO(SpaceRefiner, Initialization, ACF, Pretrain, Model, Normalizer, optimizer_config)
    elif optimizer_type == 'Bilevel':
        optimizer = Bilevel(optimizer_config)        
    return optimizer

def ConstructSelector(optimizer_config, dict = None, seed: int = 0):
    DataSelectors = {}
    
    
    for key in optimizer_config.keys():
        if key.endswith('DataSelector'):
            if optimizer_config[key] == 'None':
                DataSelectors[key] = None
            else:
                DataSelectors[key] = selector_registry[optimizer_config[key]](optimizer_config[key + 'Parameters'])
    return DataSelectors