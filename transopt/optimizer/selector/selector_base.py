

from transopt.datamanager.manager import DataManager


class SelectorBase:
    def __init__(self, config):
        self.data_manager = DataManager()

    def select(self, tasks_info, selector_info, model_info, sampler_info, acf_info, pretrain_info, refiner_info):
        raise NotImplementedError
    
    