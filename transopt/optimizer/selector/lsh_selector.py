from transopt.optimizer.selector.selector_base import SelectorBase 
from transopt.agent.registry import selector_registry

@selector_registry.register('LSH')
class LSHSelector(SelectorBase):
    def __init__(self, config):
        
        super(LSHSelector, self).__init__(config)
        
    def select(self, tasks_info):
        return super().select(tasks_info)