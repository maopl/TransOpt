from transopt.optimizer.selector.selector_base import SelectorBase 
from transopt.agent.registry import selector_registry

@selector_registry.register('rank')
class RankBasedSelector(SelectorBase):
    def __init__(self, config):
        super(RankBasedSelector, self).__init__(config)
    def select(self, tasks_info, selector_info, model_info, sampler_info, acf_info, pretrain_info, refiner_info):
        return super().select(tasks_info, selector_info, model_info, sampler_info, acf_info, pretrain_info, refiner_info)