
from transopt.agent.registry import (acf_registry, sampler_registry,
                                     selector_registry, space_refiner_registry,
                                     model_registry, pretrain_registry, normalizer_registry)
from transopt.optimizer.model.get_model import get_model
from transopt.optimizer.optimizer_base.bo import BO
from transopt.optimizer.pretrain.get_pretrain import get_pretrain



def ConstructOptimizer(optimizer_config: dict = None, seed: int = 0) -> BO:
    DataSelectors = {}
    """Create the optimizer object."""
    if optimizer_config['SpaceRefiner'] == 'None':
        SpaceRefiner = None
    else:
        SpaceRefiner = space_refiner_registry[optimizer_config['SpaceRefiner']](optimizer_config['SpaceRefinerParameters'])
        
    
    Sampler = sampler_registry[optimizer_config['Sampler']](optimizer_config['SamplerInitNum'], optimizer_config['SamplerParameters'])
    ACF = acf_registry[optimizer_config['ACF']](config = optimizer_config['ACFParameters'])
    Model = model_registry[optimizer_config['Model']](config = optimizer_config['ModelParameters'])

    if optimizer_config['Pretrain'] == 'None':
        Pretrain = None
    else:
        Pretrain = pretrain_registry[optimizer_config['Pretrain']](optimizer_config['PretrainParameters'])
        
    
    
    if optimizer_config['Normalizer'] == 'default':
        Normalizer = None
    else:
        Normalizer = normalizer_registry(optimizer_config['Normalizer'], optimizer_config['NormalizerParameters'])
    
    if optimizer_config['SpaceRefinerDataSelector'] == 'default':
        DataSelectors['SpaceRefinerDataSelector'] = None
    else:
        DataSelectors['SpaceRefinerDataSelector'] = selector_registry(optimizer_config['SpaceRefinerDataSelector'], optimizer_config['SpaceRefinerDataSelectorParameters'])
    
    if optimizer_config['SamplerDataSelector'] == 'default':
        DataSelectors['SamplerDataSelector'] = None
    else:
        DataSelectors['SamplerDataSelector'] = selector_registry(optimizer_config['SamplerDataSelector'], optimizer_config['SamplerDataSelectorParameters'])
    
    if optimizer_config['ACFDataSelector'] == 'default':
        DataSelectors['ACFDataSelector'] = None
    else:
        DataSelectors['ACFDataSelector'] = selector_registry(optimizer_config['ACFDataSelector'], optimizer_config['ACFDataSelectorParameters'])
    
    if optimizer_config['PretrainDataSelector'] == 'default':
        DataSelectors['PretrainDataSelector'] = None
    else:
        DataSelectors['PretrainDataSelector'] = selector_registry(optimizer_config['PretrainDataSelector'], optimizer_config['PretrainDataSelectorParameters'])
    
    if optimizer_config['ModelDataSelector'] == 'default':
        DataSelectors['ModelDataSelector'] = None
    else:
        DataSelectors['ModelDataSelector'] = selector_registry(optimizer_config['ModelDataSelector'], optimizer_config['ModelDataSelectorParameters'])

    if optimizer_config['NormalizerDataSelector'] == 'default':
        DataSelectors['NormalizerDataSelector'] = None
    else:
        DataSelectors['NormalizerDataSelector'] = selector_registry(optimizer_config['NormalizerDataSelector'], optimizer_config['NormalizerDataSelectorParameters'])
    
        
    
    optimizer = BO(SpaceRefiner, Sampler, ACF, Pretrain, Model, Normalizer, DataSelectors, optimizer_config)
    
    
    return optimizer
