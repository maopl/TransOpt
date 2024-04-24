import numpy as np

from transopt.agent.chat.openai_connector import (Message, OpenAIChat,
                                                  get_prompt)
from transopt.agent.config import Config, RunningConfig
from transopt.agent.registry import *
from transopt.benchmark.instantiate_problems import InstantiateProblems
from transopt.datamanager.manager import DataManager
from transopt.optimizer.construct_optimizer import ConstructOptimizer


class Services:
    def __init__(self):
        self.config = Config()
        self.running_config = RunningConfig()
        self.data_manager = DataManager()
        
        self.openai_chat = OpenAIChat(self.config)
        self.prompt = get_prompt()
        self.is_first_msg = True

        self._initialize_modules()
        

    def chat(self, user_input):
        system_message = Message(role="system", content=self.prompt)
        user_message = Message(role="user", content=user_input)
        
        if self.is_first_msg:
            response_content = self.openai_chat.get_response([system_message, user_message])
        else:
            response_content = self.openai_chat.get_response([user_message])
        
        return response_content
    
    def _initialize_modules(self):
        import transopt.benchmark.synthetic
        import transopt.optimizer.acquisition_function
        import transopt.optimizer.model
        import transopt.optimizer.pretrain
        import transopt.optimizer.refiner
        import transopt.optimizer.sampler
        
    def get_modules(self):        
        basic_info = {}
        tasks_info = []
        selector_info = [{"name": 'default'}]
        model_info = [{"name": 'default'}]
        sampler_info = [{"name": 'default'}]
        acf_info = [{"name": 'default'}]
        pretrain_info = [{"name": 'default'}]
        refiner_info = [{"name": 'default'}]
        
        # tasks information
        task_names = problem_registry.list_names()
        for name in task_names:
            if problem_registry[name].problem_type == "synthetic":
                task_info = {
                    "name": name,
                    "anyDim": True,
                    "dim": [],
                    "obj": [1],
                    "fidelity": [],
                }
            else:
                obj_num = problem_registry[name].get_objectives()
                dim = len(problem_registry[name].get_configuration_space().keys())
                task_info = {
                    "name": name,
                    "anyDim": False,
                    "dim": [dim],
                    "obj": [obj_num],
                    "fidelity": [],
                }
            tasks_info.append(task_info)
        basic_info["TasksData"] = tasks_info
        
        sampler_names = sampler_registry.list_names()
        for name in sampler_names:
            sampler_info.append({"name": name})
        basic_info["Sampler"] = sampler_info
        
        refiner_names = space_refiner_registry.list_names()
        for name in refiner_names:
            refiner_info.append({"name": name})
        basic_info["SpaceRefiner"] = refiner_info
        
        pretrain_names = pretrain_registry.list_names()
        for name in pretrain_names:
            pretrain_info.append({"name": name})
        basic_info["Pretrain"] = pretrain_info
        
        model_names = model_registry.list_names()
        for name in model_names:
            model_info.append({"name": name})
        basic_info["Model"] = model_info
        
        acf_names = acf_registry.list_names()
        for name in acf_names:
            acf_info.append({"name": name})
        basic_info["ACF"] = acf_info
        
        selector_names = selector_registry.list_names()
        for name in selector_names:
            selector_info.append({"name": name})
        basic_info["DataSelector"] = selector_info
        
        return basic_info
        
    def search_dataset(self, dataset_name, dataset_info):
        datasets_list = list(self.data_manager.get_similar_datasets(dataset_name, dataset_info))
        
        return datasets_list
    
    
    def select_dataset(self, dataset_names):
        self.running_config.set_metadata(dataset_names)
    
    def receive_tasks(self, tasks_info):
        tasks = {}
        for task in tasks_info:
            workloads = [int(item) for item in task['workloads'].split(',')]
            tasks[task["name"]] = {'budget_type': task["budget_type"],'budget': int(task['budget']), 
                                   'workloads': workloads, 'params':{'input_dim':int(task["dim"])}}

        self.running_config.set_tasks(tasks)
        return
    
    def receive_optimizer(self, optimizer_info):
        print(optimizer_info)
        optimizer = {}


        self.running_config.set_optimizer(optimizer_info)
        return
    
    def receive_metadata(self, metadata_info):
        print(metadata_info)

        self.running_config.set_metadata(metadata_info)
        return

    def get_all_tasks(self):
        all_tables = self.data_manager.db.get_table_list()
        return [self.data_manager.db.query_dataset_info(table) for table in all_tables]
    
    def run_optimize(self, seeds_info):
        seeds = [int(seed) for seed in seeds_info.split(',')]
        data_manager = DataManager()
        for seed in seeds:
            task_set = InstantiateProblems(self.running_config.tasks, seed)
            optimizer = ConstructOptimizer(self.running_config.optimizer, seed)
            
            def construct_dataset_info(task_set):
                dataset_info = {}
                dataset_info['variables'] = [{"name": var.name, "type": var.type, "range": var.range} for var_name, var in task_set.get_cur_searchspace_info().items()]
                dataset_info['objectives'] = [{"name": name, "type": type} for name, type in task_set.get_curobj_info().items()]
                dataset_info['fidelities'] = [{"name": var.name, "type": var.type, "range": var.range} for var_name, var in task_set.get_cur_fidelity_info().items()]

                return dataset_info
            
            while (task_set.get_unsolved_num()):
                search_space = task_set.get_cur_searchspace()
                dataset_info = construct_dataset_info(task_set)
                
                data_manager.db.create_table(task_set.get_curname(), dataset_info)
                optimizer.link_task(task_name=task_set.get_curname(), search_sapce=search_space)
                optimizer.set_metadata()
                optimizer.search_space_refine()
                samples = optimizer.sample_initial_set()
                parameters = [search_space.map_to_design_space(sample) for sample in samples]
                observations = task_set.f(parameters) 
                
                # data_manager.db.insert_data(task_set.get_curname(), parameters.update(observations))
                
                while (task_set.get_rest_budget()):
                    suggested_sample = self.suggest()
                    parameters = search_space.map_to_design_space(suggested_sample)
                    observations = task_set.f(parameters)
                    # data_manager.db.insert_data(task_set.get_curname(), parameters.update(observations))
                    
                    optimizer.observe(search_space.map_from_design_space(suggested_sample), observations)
                    # if self.verbose:
                    #     self.visualization(testsuits, suggested_sample)
                task_set.roll()
                

