
from transopt.agent.registry import (acf_registry, sampler_registry,
                                     selector_registry, space_refiner_registry,
                                     model_registry, pretrain_registry)
from transopt.optimizer.model.get_model import get_model
from transopt.optimizer.optimizer_base.bo import BO
from transopt.optimizer.pretrain.get_pretrain import get_pretrain



def ConstructOptimizer(optimizer_config: dict = None, seed: int = 0) -> BO:
    """Create the optimizer object."""
    if optimizer_config['SpaceRefiner'] == 'default':
        SpaceRefiner = None
    else:
        SpaceRefiner = space_refiner_registry[optimizer_config['SpaceRefiner']](optimizer_config['SpaceRefinerParameters'])
    
    if optimizer_config['Sampler'] == 'default':
        Sampler = sampler_registry['lhs'](config={})
    else:
        Sampler = sampler_registry[optimizer_config['Sampler']](optimizer_config['SamplerParameters'])
        
    
    if optimizer_config['ACF'] == 'default':
        ACF = acf_registry['EI'](config={})
    else:
        ACF = acf_registry[optimizer_config['ACF']](config = optimizer_config['ACFParameters'])
        
    if optimizer_config['Pretrain'] == 'default':
        Pretrain = None
    else:
        Pretrain = pretrain_registry[optimizer_config['Pretrain']](optimizer_config['PretrainParameters'])
        
    
    if optimizer_config['Model'] == 'default':
        Model = model_registry['GP'](config={'kernel': 'RBF'})
    else:
        Model = model_registry[optimizer_config['Model']](optimizer_config['ModelParameters'])
    
    
    if optimizer_config['DataSelector'] == 'default':
        DataSelector = None
    else:
        DataSelector = selector_registry(optimizer_config['DataSelector'], optimizer_config['DataSelectorParameters'])
    
    optimizer = BO(SpaceRefiner, Sampler, ACF, Pretrain, Model, DataSelector, optimizer_config)
    
    
    return optimizer
