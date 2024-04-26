from agent.chat.openai_chat import Message, OpenAIChat
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

        self.openai_chat = OpenAIChat(
            api_key=self.config.OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            base_url=self.config.OPENAI_URL,
        )

        self._initialize_modules()

    def chat(self, user_input):
        response_content = self.openai_chat.get_response(user_input)
        return response_content

    def _initialize_modules(self):
        import transopt.benchmark.synthetic
        # import transopt.benchmark.CPD
        import transopt.optimizer.acquisition_function
        import transopt.optimizer.model
        import transopt.optimizer.pretrain
        import transopt.optimizer.refiner
        import transopt.optimizer.sampler

    def get_modules(self):
        basic_info = {}
        tasks_info = []
        selector_info = [{"name": "default"}]
        model_info = [{"name": "default"}]
        sampler_info = [{"name": "default"}]
        acf_info = [{"name": "default"}]
        pretrain_info = [{"name": "default"}]
        refiner_info = [{"name": "default"}]
        normalizer_info = [{"name": "default"}]

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
        
        normalizer_names = selector_registry.list_names()
        for name in normalizer_names:
            normalizer_info.append({"name": name})
        basic_info["Normalizer"] = normalizer_info

        return basic_info

    def search_dataset(self, search_method, dataset_name, dataset_info):
        if search_method == 'Fuzzy':
            datasets_list = {"isExact": False, 
                             "datasets": list(self.data_manager.search_datasets_by_name(dataset_name))}
        elif search_method == 'Hash':
            dataset_detail_info = self.data_manager.get_dataset_info(dataset_name)
            if dataset_detail_info:
                datasets_list = {"isExact": True, "datasets": dataset_detail_info['additional_config']}
            else:
                raise ValueError("Dataset not found")
        elif search_method == 'LSH':
            datasets_list = {"isExact": False, 
                             "datasets":list(self.data_manager.search_similar_datasets(dataset_name, dataset_info))}
            
        else:
            raise ValueError("Invalid search method")

        return datasets_list

    def select_dataset(self, dataset_names):
        self.running_config.set_metadata(dataset_names)

    def receive_tasks(self, tasks_info):
        tasks = {}
        for task in tasks_info:
            workloads = [int(item) for item in task["workloads"].split(",")]
            tasks[task["name"]] = {
                "budget_type": task["budget_type"],
                "budget": int(task["budget"]),
                "workloads": workloads,
                "params": {"input_dim": int(task["dim"])},
            }

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
    
    def construct_dataset_info(self, task_set, running_config):
        dataset_info = {}
        dataset_info["variables"] = [
            {"name": var.name, "type": var.type, "range": var.range}
            for var_name, var in task_set.get_cur_searchspace_info().items()
        ]
        dataset_info["objectives"] = [
            {"name": name, "type": type}
            for name, type in task_set.get_curobj_info().items()
        ]
        dataset_info["fidelities"] = [
            {"name": var.name, "type": var.type, "range": var.range}
            for var_name, var in task_set.get_cur_fidelity_info().items()
        ]

        dataset_info['additional_config'] = {
            "name": task_set.get_curname(),
            "dim": len(dataset_info["variables"]),
            "obj": len(dataset_info["objectives"]),
            "fidelity": ', '.join([d['name'] for d in dataset_info["fidelities"] if 'name' in d]) if len(dataset_info["fidelities"]) == 0 else '',
            "workloads": task_set.get_curtworkload(),
            "budget_type": task_set.get_cur_budgettype(),
            "budget": task_set.get_cur_budget(),
            "seeds": task_set.get_curseed(),
            "SpaceRefiner": running_config.optimizer['SpaceRefiner'],
            "Sampler": running_config.optimizer['Sampler'],
            "Pretrain": running_config.optimizer['Pretrain'],
            "Model": running_config.optimizer['Model'],
            "ACF": running_config.optimizer['ACF'],
            "DatasetSelector": running_config.optimizer['DataSelector'],
            "Normalizer": running_config.optimizer['Normalizer'],
            "datasets": running_config.metadata if running_config.metadata else [],
        }
        return dataset_info
    
    def get_metadata(self):
        if self.running_config.metadata:
            metadata = {}
            metadata_info = {}
            for dataset_name in self.running_config.metadata:
                metadata[dataset_name] = self.data_manager.db.select_data(dataset_name)
                metadata_info[dataset_name] = self.data_manager.db.query_dataset_info(dataset_name)
            return metadata, metadata_info
        else:
            return None
                

    def run_optimize(self, seeds_info):
        seeds = [int(seed) for seed in seeds_info.split(",")]
        for seed in seeds:
            task_set = InstantiateProblems(self.running_config.tasks, seed)
            optimizer = ConstructOptimizer(self.running_config.optimizer, seed)
            
            def save_data(parameters, observations):
                data = [{} for i in range(len(parameters))]
                [data[i].update(parameters[i]) for i in range(len(parameters))]
                [data[i].update(observations[i]) for i in range(len(parameters))]
                [data[i].update({'batch':iterations}) for i in range(len(parameters))]
                self.data_manager.db.insert_data(task_set.get_curname(), data)
            
            try:
                while (task_set.get_unsolved_num()):
                    iterations = 0
                    search_space = task_set.get_cur_searchspace()
                    dataset_info = self.construct_dataset_info(task_set, self.running_config)
                    
                    self.data_manager.db.create_table(task_set.get_curname(), dataset_info, overwrite=True)
                    optimizer.link_task(task_name=task_set.get_curname(), search_sapce=search_space)
                    metadata, metadata_info = self.get_metadata()
                    optimizer.search_space_refine(metadata)
                    samples = optimizer.sample_initial_set(metadata)
                    parameters = [search_space.map_to_design_space(sample) for sample in samples]
                    observations = task_set.f(parameters)
                    save_data(parameters, observations)
                    
                    optimizer.observe(samples, observations)
                    
                    #Pretrain
                    optimizer.meta_fit(metadata, metadata_info)
            
                    while (task_set.get_rest_budget()):
                        optimizer.fit()
                        suggested_samples = optimizer.suggest()
                        parameters = [search_space.map_to_design_space(sample) for sample in suggested_samples]
                        observations = task_set.f(parameters)
                        save_data(parameters, observations)
                        
                        optimizer.observe(suggested_samples, observations)
                        iterations += 1
                        
                        print("Seed: ", seed, "Task: ", task_set.get_curname(), "Iteration: ", iterations)
                        # if self.verbose:
                        #     self.visualization(testsuits, suggested_sample)
                    task_set.roll()
            except Exception as e:
                raise e

    def get_report_charts(self, task_name):
        all_data = self.data_manager.db.select_data(task_name)

        table_info = self.data_manager.db.query_dataset_info(task_name)
        objectives = table_info["objectives"]

        obj = objectives[0]["name"]
        obj_type = objectives[0]["type"]

        obj_data = [data[obj] for data in all_data]

        ret = {
            "RadarData": {
                "indicator": [
                    {"name": "F1 score", "max": 1},
                    {"name": "Accuracy", "max": 1},
                    {"name": "Recall", "max": 1},
                    {"name": "Root Mean Squared Error", "max": 10},
                    {"name": "AUC-ROC ", "max": 1},
                    {"name": "BOA-AUC ", "max": 1},
                ],
                "data": [{"value": [0.8, 0.95, 0.5, 2.42639, 0.7, 0.8]}],
            },
            "BarData": [
                {"value": 0.25, "name": "Learning_rate"},
                {"value": 0.36, "name": "Neuron number"},
                {"value": 0.76, "name": "Layer number"},
                {"value": 0.12, "name": "Block number"},
                {"value": 0.54, "name": "Weight decay"},
                {"value": 0.72, "name": "Momentum"},
            ],
            "ScatterData": {
                "cluster1": [
                    [10.0, 8.04],
                    [8.07, 6.95],
                    [13.0, 7.58],
                    [9.05, 8.81],
                    [11.0, 8.33],
                    [14.0, 7.66],
                    [13.4, 6.81],
                    [10.0, 6.33],
                ],
                "cluster2": [
                    [14.0, 8.96],
                    [12.5, 6.82],
                    [9.15, 7.2],
                    [11.5, 7.2],
                    [3.03, 4.23],
                    [12.2, 7.83],
                    [2.02, 4.47],
                    [1.05, 3.33],
                ],
                "cluster3": [
                    [4.05, 4.96],
                    [6.03, 7.24],
                    [12.0, 6.26],
                    [12.0, 8.84],
                    [7.08, 5.82],
                    [5.02, 5.68],
                ],
            },
        }
        ret.update(self.construct_trajectory_data(task_name, obj_data, obj_type))

        return ret

    def construct_trajectory_data(self, name, obj_data, obj_type="minimize"):
        # Initialize the list to store trajectory data and the best value seen so far
        trajectory = []
        best_value = float("inf") if obj_type == "minimize" else -float("inf")
        best_values_so_far = []

        # Loop through each function evaluation
        for index, current_value in enumerate(obj_data, start=1):
            # Update the best value based on the objective type
            if obj_type == "minimize":
                if current_value < best_value:
                    best_value = current_value
            else:  # maximize
                if current_value > best_value:
                    best_value = current_value

            # Append the best value observed so far to the list
            best_values_so_far.append(best_value)
            trajectory.append({"FEs": index, "y": best_value})

        uncertainty = []
        for data_point in trajectory:
            base_value = data_point["y"]
            uncertainty_range = [base_value, base_value]
            uncertainty.append({"FEs": data_point["FEs"], "y": uncertainty_range})

        trajectory_data = {
            "name": name,
            "average": trajectory,
            "uncertainty": uncertainty,
        }

        return {"TrajectoryData": [trajectory_data]}
