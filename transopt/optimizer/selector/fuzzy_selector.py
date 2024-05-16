from transopt.agent.registry import selector_registry
from transopt.optimizer.selector.selector_base import SelectorBase


@selector_registry.register("Fuzzy")
class FuzzySelector(SelectorBase):
    def __init__(self, config):
        super(FuzzySelector, self).__init__(config)

    def fetch_data(self, tasks_info):
        task_name = tasks_info["additional_config"]["problem_name"]
        dimensions = tasks_info["additional_config"]["dimensions"]
        conditions = {
            "task_name": task_name,
            "dimensions": dimensions,
        }

        datasets_list = self.data_manager.db.search_tables_by_metadata(conditions)
        metadata = {
            dataset_name: self.data_manager.db.select_data(dataset_name)
            for dataset_name in datasets_list
        }
        metadata_info = {
            dataset_name: self.data_manager.db.query_dataset_info(dataset_name)
            for dataset_name in datasets_list
        }

        return metadata, metadata_info
