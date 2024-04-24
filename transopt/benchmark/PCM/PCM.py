import os
import re
import math
import logging
import numpy as np
from typing import Union, Dict

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
        protein_list = []
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PCM-Protein-Structure-Prediction")
        for filename in os.listdir(os.path.join(file_path, "protein_structure")):
            if filename.endswith(".seq"):
                protein_list.append(filename)
        self.selected_protein = protein_list[workload]
        pass


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
        
        return {self.objective_info[0]: float(energy), "info": {"fidelity": fidelity}}
    
    def get_configuration_space(self) -> SearchSpace:
        variables = [Categorical(
            f'aa{i}',
            ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        ) for i in range(11)]
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
    
pcm = PCM("PCM", "FEs", 100, 0, 0)