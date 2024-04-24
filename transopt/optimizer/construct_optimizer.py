
from transopt.optimizer.acquisition_function.get_acf import get_acf
from transopt.optimizer.model.get_model import get_model
from transopt.optimizer.optimizer_base.bo import BO
from transopt.optimizer.pretrain.get_pretrain import get_pretrain
from transopt.optimizer.refiner.get_refiner import get_refiner
from transopt.optimizer.sampler.get_sampler import get_sampler

from transopt.agent.registry import acf_registry


def ConstructOptimizer(optimizer_config: dict = None, seed: int = 0) -> BO:
    """Create the optimizer object."""
    if optimizer_config['SpaceRefiner'] == 'default':
        SpaceRefiner = None
    else:
        SpaceRefiner = get_refiner(optimizer_config['SpaceRefiner'], optimizer_config['SpaceRefinerParameters'])
    
    if optimizer_config['Sampler'] == 'default':
        Sampler = get_sampler('lhs', config={})
    else:
        Sampler = get_sampler(optimizer_config['Sampler'], optimizer_config['SamplerParameters'])
        
    
    if optimizer_config['ACF'] == 'default':
        ACF = acf_registry['EI'](config={})
    else:
        ACF = acf_registry[optimizer_config['ACF']](config = optimizer_config['ACFParameters'])
        
    if optimizer_config['Pretrain'] == 'default':
        Pretrain = None
    else:
        Pretrain = get_pretrain(optimizer_config['Pretrain'], optimizer_config['PretrainParameters'])
        
    
    if optimizer_config['Model'] == 'default':
        Model = get_model('GP', config={'kernel': 'RBF'})
    else:
        Model = get_model(optimizer_config['Model'], optimizer_config['ModelParameters'])
    
    optimizer = BO(SpaceRefiner, Sampler, ACF, Pretrain, Model, optimizer_config)
    
    
    return optimizer
