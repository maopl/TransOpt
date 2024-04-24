import os
import json
import numpy as np
from typing import Union, Dict
from PCM.energy.energy_nofile import *
from numpy.random.mtrand import RandomState as RandomState
from transopt.space.variable import *
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace


@problem_registry.register("PCM")
class PCM(NonTabularProblem):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        # Select protein based on workload. The workload is an integer between 0 and 38.
        protein_list = []
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PCM")
        for filename in os.listdir(os.path.join(file_path, "protein_structure")):
            if filename.endswith(".seq"):
                protein_list.append(filename)
        self.selected_protein = protein_list[workload]

        # modify the config file
        with open(os.path.join(file_path, 'config/config_PCM_1ZDD.json'), 'r') as file:
            data = json.load(file)
        data["protein_params"]["name"] = self.selected_protein.split(".")[0]
        data["protein_params"]["second_struct_file"] = os.path.join(file_path,"protein_structure/") + self.selected_protein
        with open(os.path.join(file_path, 'config/config_PCM.json'), 'w') as file:
            json.dump(data, file, indent=4)


        # create a Protein object
        self.energy_temp_save_path = os.path.join(file_path, 'logs/log/enegy_temp_save')
        if not os.path.exists(self.energy_temp_save_path):
            os.makedirs(self.energy_temp_save_path)
        
        config_file = open(os.path.join(file_path,'config/config_PCM.json'), 'r').read()
        config = json.loads(config_file)
        protein_config_file = open(os.path.join(file_path,'config/protein_config.json'), 'r').read()
        protein_config = json.loads(protein_config_file)
        energy_config_file = open(os.path.join(file_path,'config/energy_config.json'), 'r').read()
        self.energy_config = json.loads(energy_config_file)

        self.second_struct_file_path = config['protein_params']['second_struct_file']
        protein_status = config['protein_params']['status']
        self.protein_name = config['protein_params']['name']
        self.root = config['paths']['root']
        pop_size = config['algo_params']['pop_size']
        num_obj = config['energy_params']['number_objective']
        self.max_thread = config['energy_params']['max_thread']
        coder = Coding(protein_config, protein_status)

        self.proteins = []
        new_protein = Protein(num_obj, protein_status, coder)
        protein = coder.decoder_from_seq(self.second_struct_file_path, new_protein)
        self.max_angles, self.min_angles = protein.get_angles_field()
        self.proteins.append(protein)

        super(PCM, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
    
    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs
    ) -> Dict:
        angels = list(configuration.values())
        self.proteins[0].update_angle_from_view(angels)
        energy = Energy(self.energy_config, self.root, self.energy_temp_save_path, self.protein_name, self.second_struct_file_path, self.max_thread, self.proteins[0])
        energy.calculate_energy(self.proteins)
        self.__clear_folder(self.energy_temp_save_path)
        proteins_energy = self.proteins[0].obj
        return {self.objective_info[0]: float(proteins_energy[0]),
                self.objective_info[1]: float(proteins_energy[1]),
                self.objective_info[2]: float(proteins_energy[2]),
                self.objective_info[3]: float(proteins_energy[3]),
                "info": {"fidelity": fidelity}}
    
    def get_configuration_space(self) -> SearchSpace:
        variables = [Continuous(f'angle{i}', [self.min_angles[i],self.max_angles[i]]) for i in range(len(max_angles))]
        ss = SearchSpace(variables)
        return ss

    def get_fidelity_space(self) -> FidelitySpace:
        fs = FidelitySpace([])
        return fs
    
    def get_objectives(self) -> list:
        return ["bond_energy", "dDFIRE", "Rosetta", "RWplus"]
    
    def get_problem_type(self) -> str:
        return "CPD"
    
    def get_meta_information(self) -> Dict:
        return {}
    
    def __clear_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    