
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
    if optimizer_config['SpaceRefiner'] == 'None':
        SpaceRefiner = None
    else:
        if 'SpaceRefinerParameters' not in optimizer_config:
            optimizer_config['SpaceRefinerParameters'] = {}
        SpaceRefiner = space_refiner_registry[optimizer_config['SpaceRefiner']](optimizer_config['SpaceRefinerParameters'])
        
    
    Sampler = sampler_registry[optimizer_config['Sampler']](optimizer_config['SamplerInitNum'], optimizer_config['SamplerParameters'])
    ACF = acf_registry[optimizer_config['ACF']](config = optimizer_config['ACFParameters'])

    # Model = model_registry[optimizer_config['Model']](config = optimizer_config['ModelParameters'])
    Model = model_registry[optimizer_config['Model']]()

    if optimizer_config['Pretrain'] == 'None':
        Pretrain = None
    else:
        Pretrain = pretrain_registry[optimizer_config['Pretrain']](optimizer_config['PretrainParameters'])
        
    
    
    if optimizer_config['Normalizer'] == 'None':
        Normalizer = None
    else:
        Normalizer = normalizer_registry[optimizer_config['Normalizer']](optimizer_config['NormalizerParameters'])
        
    
    ''' Bugee original code. No 'Optimizer' in optimizer_config
    
    if optimizer_config['Optimizer'] == 'BO':
        optimizer = BO(SpaceRefiner, Sampler, ACF, Pretrain, Model, Normalizer, optimizer_config)
    elif optimizer_config['Optimizer'] == 'Bilevel':
        optimizer = Bilevel(optimizer_config)

    '''
    # Just for test.
    optimizer_type = optimizer_config.get('Optimizer', 'BO')
    if optimizer_type == 'BO':
        optimizer = BO(SpaceRefiner, Sampler, ACF, Pretrain, Model, Normalizer, optimizer_config)
    elif optimizer_type == 'Bilevel':
        optimizer = Bilevel(optimizer_config)        
    return optimizer

def ConstructSelector(optimizer_config, dict = None, seed: int = 0):
    DataSelectors = {}
    
    
    # if optimizer_config['SpaceRefinerDataSelector'] == 'None':
    #     DataSelectors['SpaceRefinerDataSelector'] = None
    # else:
    #     DataSelectors['SpaceRefinerDataSelector'] = selector_registry(optimizer_config['SpaceRefinerDataSelector'], optimizer_config['SpaceRefinerDataSelectorParameters'])
    
    # if optimizer_config['SamplerDataSelector'] == 'None':
    #     DataSelectors['SamplerDataSelector'] = None
    # else:
    #     DataSelectors['SamplerDataSelector'] = selector_registry(optimizer_config['SamplerDataSelector'], optimizer_config['SamplerDataSelectorParameters'])
    
    # if optimizer_config['ACFDataSelector'] == 'None':
    #     DataSelectors['ACFDataSelector'] = None
    # else:
    #     DataSelectors['ACFDataSelector'] = selector_registry(optimizer_config['ACFDataSelector'], optimizer_config['ACFDataSelectorParameters'])
    
    # if optimizer_config['PretrainDataSelector'] == 'None':
    #     DataSelectors['PretrainDataSelector'] = None
    # else:
    #     DataSelectors['PretrainDataSelector'] = selector_registry(optimizer_config['PretrainDataSelector'], optimizer_config['PretrainDataSelectorParameters'])
    
    # if optimizer_config['ModelDataSelector'] == 'None':
    #     DataSelectors['ModelDataSelector'] = None
    # else:
    #     DataSelectors['ModelDataSelector'] = selector_registry(optimizer_config['ModelDataSelector'], optimizer_config['ModelDataSelectorParameters'])

    # if optimizer_config['NormalizerDataSelector'] == 'None':
    #     DataSelectors['NormalizerDataSelector'] = None
    # else:
    #     DataSelectors['NormalizerDataSelector'] = selector_registry(optimizer_config['NormalizerDataSelector'], optimizer_config['NormalizerDataSelectorParameters'])
    
    
    for key in optimizer_config.keys():
        if key.endswith('DataSelector'):
            if optimizer_config[key] == 'None':
                DataSelectors[key] = None
            else:
                DataSelectors[key] = selector_registry[optimizer_config[key]](optimizer_config[key + 'Parameters'])
    return DataSelectors