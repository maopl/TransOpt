import time
import numpy as np
from multiprocessing import Process, Manager

from agent.chat.openai_chat import Message, OpenAIChat

from transopt.agent.config import Config, RunningConfig
from transopt.agent.registry import *
from transopt.benchmark.instantiate_problems import InstantiateProblems
from transopt.datamanager.manager import DataManager, Database
from transopt.optimizer.construct_optimizer import ConstructOptimizer
from transopt.utils.log import logger
from analysis.mds import FootPrint

class Services:
    def __init__(self, task_queue, result_queue, lock):
        self.config = Config()
        self.running_config = RunningConfig()
        
        # DataManager for general tasks, not specific optimization tasks
        self.data_manager = DataManager()

        self.openai_chat = OpenAIChat(
            api_key=self.config.OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            base_url=self.config.OPENAI_URL,
            data_manager= self.data_manager
        )

        self._initialize_modules()
        self.process_info = Manager().dict()

    def chat(self, user_input):
        response_content = self.openai_chat.get_response(user_input)
        return response_content

    def _initialize_modules(self):
        # import transopt.benchmark.CPD
        import transopt.benchmark.HPO
        import transopt.benchmark.synthetic
        # import transopt.benchmark.CPD
        import transopt.benchmark.HPO
        import transopt.optimizer.acquisition_function
        import transopt.optimizer.model
        import transopt.optimizer.pretrain
        import transopt.optimizer.refiner
        import transopt.optimizer.sampler

    def get_modules(self):
        basic_info = {}
        tasks_info = []
        selector_info = []
        model_info = []
        sampler_info = []
        acf_info = []
        pretrain_info = []
        refiner_info = []
        normalizer_info = [{'name':'default'}]

        # tasks information
        task_names = problem_registry.list_names()
        for name in task_names:
            if problem_registry[name].problem_type == "synthetic":
                num_obj = problem_registry[name].num_objectives
                num_var = problem_registry[name].num_variables
                task_info = {
                    "name": name,
                    "problem_type": "synthetic",
                    "anyDim": "True",
                    'num_vars': [],
                    "num_objs": [1],
                    "workloads": [],
                    "fidelity": [],
                }
            else:
                num_obj = problem_registry[name].num_objectives
                num_var = problem_registry[name].num_variables
                fidelity = problem_registry[name].fidelity
                workloads = problem_registry[name].workloads
                problem_type = problem_registry[name].problem_type
                task_info = {
                    "name": name,
                    "problem_type": problem_type,
                    "anyDim": False,
                    "num_vars": [num_var],
                    "num_objs": [num_obj],
                    "workloads": workloads,
                    "fidelity": [fidelity],
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
   
    def convert_metadata(self, conditions):
        type_map = {
            "NumVars": int,
            "NumObjs": int,
            "Workload": int,
            "Seed": int,
            # Add other fields as necessary
        }
        converted_conditions = {}
        for key, value in conditions.items():
            if key in type_map:
                try:
                    # Convert the value according to its expected type
                    if type_map[key] == int:
                        converted_conditions[key] = int(value)
                    elif type_map[key] == float:
                        converted_conditions[key] = float(value)
                    elif type_map[key] == bool:
                        converted_conditions[key] = value.lower() in ['true', '1', 't', 'yes', 'y']
                    else:
                        converted_conditions[key] = value  # Assume string or no conversion needed
                except ValueError:
                    raise ValueError(f"Invalid value for {key}: {value}")
            else:
                # If no specific type is expected, assume string
                converted_conditions[key] = value

        return converted_conditions
 
    def comparision_search(self, conditions):
        conditions = {k: v for k, v in conditions.items() if v}
        conditions = self.convert_metadata(conditions)
        
        key_map = {
            "TaskName": "problem_name",
            "NumVars": "dimensions",
            "NumObjs": "objectives",
            "Fidelity": "fidelities",
            "Workload": "workloads",
            "Seed": "seeds",
            "Refiner": "space_refiner",
            "Sampler": "sampler",
            "Pretrain": "pretrain",
            "Model": "model",
            "ACF": "acf",
            "Normalizer": "normalizer"
        }
        
        # change key in conditions to match the key in database
        conditions = {key_map[k]: v for k, v in conditions.items() if k in key_map}
        
        return self.data_manager.db.search_tables_by_metadata(conditions)
    
    def set_metadata(self, dataset_names):
        self.running_config.set_metadata(dataset_names)

    def receive_tasks(self, tasks_info):
        tasks = {}
        for task in tasks_info:
            workloads = [int(item) for item in task["workloads"].split(",")]
            tasks[task["name"]] = {
                "budget_type": task["budget_type"],
                "budget": int(task["budget"]),
                "workloads": workloads,
                "params": {"input_dim": int(task["num_vars"])},
            }

        self.running_config.set_tasks(tasks)
        return

    def receive_optimizer(self, optimizer_info):

        self.running_config.set_optimizer(optimizer_info)
        return

    def receive_metadata(self, metadata_info):
        print(metadata_info)

        self.running_config.set_metadata(metadata_info)
        return

    def get_all_datasets(self):
        all_tables = self.data_manager.db.get_table_list()
        return [self.data_manager.db.query_dataset_info(table) for table in all_tables]
    
    def get_experiment_datasets(self):
        experiment_tables = self.data_manager.db.get_table_list()
        return [(experiment_tables[table_id],self.data_manager.db.query_dataset_info(table)) for table_id, table in enumerate(experiment_tables)] 
   
    def construct_dataset_info(self, task_set, running_config, seed):
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

        # Simplify dataset name construction
        timestamp = int(time.time())
        dataset_name = f"{task_set.get_curname()}_w{task_set.get_cur_workload()}_s{seed}_{timestamp}"

        dataset_info['additional_config'] = {
            "problem_name": task_set.get_curname(),
            "dim": len(dataset_info["variables"]),
            "obj": len(dataset_info["objectives"]),
            "fidelity": ', '.join([d['name'] for d in dataset_info["fidelities"] if 'name' in d]) if dataset_info["fidelities"] else '',
            "workloads": task_set.get_cur_workload(),
            "budget_type": task_set.get_cur_budgettype(),
            "budget": task_set.get_cur_budget(),
            "seeds": seed,
            "SpaceRefiner": running_config.optimizer['SpaceRefiner'],
            "Sampler": running_config.optimizer['Sampler'],
            "Pretrain": running_config.optimizer['Pretrain'],
            "Model": running_config.optimizer['Model'],
            "ACF": running_config.optimizer['ACF'],
            "Normalizer": running_config.optimizer['Normalizer'],
            "DatasetSelector": f"SpaceRefiner-{running_config.optimizer['SpaceRefinerDataSelector']}, \
                Sampler - {running_config.optimizer['SamplerDataSelector']}, \
                Pretrain - {running_config.optimizer['PretrainDataSelector']}, \
                Model - {running_config.optimizer['ModelDataSelector']}, \
                ACF-{running_config.optimizer['ACFDataSelector']}, \
                Normalizer - {running_config.optimizer['NormalizerDataSelector']}",
            "metadata": running_config.metadata if running_config.metadata else [],
        }

        return dataset_info, dataset_name
 
    def get_metadata(self, module_name):
        if len(self.running_config.metadata[module_name]):
            metadata = {}
            metadata_info = {}
            for dataset_name in self.running_config.metadata[module_name]:
                metadata[dataset_name] = self.data_manager.db.select_data(dataset_name)
                metadata_info[dataset_name] = self.data_manager.db.query_dataset_info(dataset_name)
            return metadata, metadata_info
        else:
            return None, None
    
    def save_data(self, dataset_name, parameters, observations, iteration):
        data = [{} for i in range(len(parameters))]
        [data[i].update(parameters[i]) for i in range(len(parameters))]
        [data[i].update(observations[i]) for i in range(len(parameters))]
        [data[i].update({'batch':iteration}) for i in range(len(parameters))]
        self.data_manager.db.insert_data(dataset_name, data)
    
    def remove_dataset(self, dataset_name):
        if isinstance(dataset_name, str):
            self.data_manager.db.remove_table(dataset_name)
        elif isinstance(dataset_name, list):
            for name in dataset_name:
                self.data_manager.db.remove_table(name)
        else:
            raise ValueError("Invalid dataset name")

    def run_optimize(self, seeds):
        # Create a separate process for each seed
        process_list = []
        for seed in seeds:
            p = Process(target=self._run_optimize_process, args=(seed,))
            process_list.append(p)
            p.start()
        
        for p in process_list:
            p.join()
            
    def _run_optimize_process(self, seed):
        # Each process constructs its own DataManager
        import os
        pid = os.getpid()
        self.process_info[pid] = {'status': 'running', 'seed': seed, 'task': None, 'iteration': 0, 'dataset_name': None}
        logger.info(f"Start process #{pid}")

        # Instantiate problems and optimizer
        task_set = InstantiateProblems(self.running_config.tasks, seed)
        optimizer = ConstructOptimizer(self.running_config.optimizer, seed)

        while (task_set.get_unsolved_num()):
            self.process_info[pid]['task'] = task_set.get_curname()
            search_space = task_set.get_cur_searchspace()
            
            dataset_info, dataset_name = self.construct_dataset_info(task_set, self.running_config, seed=seed)
            self.process_info[pid]['dataset_name'] = dataset_name
            
            self.data_manager.db.create_table(dataset_name, dataset_info, overwrite=True)
            optimizer.link_task(task_name=task_set.get_curname(), search_space=search_space)
                    
            metadata, metadata_info = self.get_metadata('SpaceRefiner')
            optimizer.search_space_refine(metadata, metadata_info)
                    
            metadata, metadata_info = self.get_metadata('Sampler')
            samples = optimizer.sample_initial_set(metadata, metadata_info)
                    
            parameters = [search_space.map_to_design_space(sample) for sample in samples]
            observations = task_set.f(parameters)
            self.save_data(dataset_name, parameters, observations, self.process_info[pid]['iteration'])
                    
            optimizer.observe(samples, observations)
                    
            # Pretrain
            metadata, metadata_info = self.get_metadata('Model')
            optimizer.meta_fit(metadata, metadata_info)
            
            while (task_set.get_rest_budget()):
                optimizer.fit()
                suggested_samples = optimizer.suggest()
                parameters = [search_space.map_to_design_space(sample) for sample in suggested_samples]
                observations = task_set.f(parameters)
                if observations is None:
                    break
                self.save_data(dataset_name, parameters, observations, self.process_info[pid]['iteration'])
                        
                optimizer.observe(suggested_samples, observations)
                self.process_info[pid]['iteration'] += 1
                logger.info(f"PID {pid}: Seed {seed}, Task {task_set.get_curname()}, Iteration {self.process_info[pid]['iteration']}")
            task_set.roll()
        
        self.process_info[pid]['status'] = 'completed'
    
    def get_all_process_info(self):
        return dict(self.process_info)
    
    def get_box_plot_data(self, task_names):
        all_data = {}
        for group_id, group in enumerate(task_names):
            all_data[str(group_id)] = []
            for task_name in group:
                data = self.data_manager.db.select_data(task_name)
                table_info = self.data_manager.db.query_dataset_info(task_name)
                objectives = table_info["objectives"]
                obj = objectives[0]["name"]
                best_obj = min([d[obj] for d in data])
                
                all_data[str(group_id)].append(best_obj)

        return all_data
    
    def get_report_charts(self, task_name):
        all_data = self.data_manager.db.select_data(task_name)

        table_info = self.data_manager.db.query_dataset_info(task_name)
        objectives = table_info["objectives"]
        ranges = [tuple(var['range']) for var in table_info["variables"]]

        obj = objectives[0]["name"]
        obj_type = objectives[0]["type"]

        obj_data = [data[obj] for data in all_data]
        var_data = [[data[var["name"]] for var in table_info["variables"]] for data in all_data]
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
        }
        ret.update(self.construct_footprint_data(task_name, var_data, ranges))
        ret.update(self.construct_trajectory_data(task_name, obj_data, obj_type))

        return ret

    def construct_footprint_data(self, name, var_data, ranges):
        # Initialize the list to store trajectory data and the best value seen so far
        fp = FootPrint(var_data, ranges)
        fp.calculate_distances()
        fp.get_mds()
        scatter_data = {'parameters': fp._reduced_data[:len(fp.X)], 'boundary': fp._reduced_data[len(fp.X):]}

        return {"ScatterData": scatter_data}
    
    def construct_statistic_trajectory_data(self, task_names):
        all_data = []
        for group_id, group in enumerate(task_names):
            min_data = {'name': f'Algorithm{group_id + 1}', 'average': [], 'uncertainty': []}
            res = []
            max_length = 0
            for task_name in group:
                data = self.data_manager.db.select_data(task_name)
                table_info = self.data_manager.db.query_dataset_info(task_name)
                objectives = table_info["objectives"]
                obj = objectives[0]["name"]
                obj_data = [d[obj] for d in data]
                acc_obj_data = np.minimum.accumulate(obj_data).flatten().tolist()
                res.append(acc_obj_data)
                if len(acc_obj_data) > max_length:
                    max_length = len(acc_obj_data)

            # 计算每个点的中位数和标准差
            for i in range(max_length):
                current_data = [r[i] for r in res if i < len(r)]
                median = np.median(current_data)
                std = np.std(current_data)
                min_data['average'].append({'FEs': i + 1, 'y': median})
                min_data['uncertainty'].append({'FEs': i + 1, 'y': [median - std, median + std]})

            all_data.append(min_data)

        return all_data
        
    
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
