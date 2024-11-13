import os

import numpy as np
import pandas as pd
import RNA

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *


def get_structures():
    base_dir = os.path.dirname(__file__)
    unique_file = os.path.join(base_dir, "inverse_rna_folding_unique_dotbracket.pkl.gz")
    
    df = pd.read_pickle(unique_file)
    target_structures = df['dotbracket'].tolist()
    workloads = list(range(len(target_structures)))
    
    return target_structures, workloads


_all_target_structures, _workloads = get_structures()


@problem_registry.register("RNAInverseDesign")
class RNAInverseDesign(NonTabularProblem):
    problem_type = "RNAInverseDesign"
    fidelity = None
    target_structures = _all_target_structures  # Internal storage for lazy-loaded target structures
    workloads = _workloads          # Internal storage for lazy-loaded workloads
    num_variables = []          # Based on workload
    num_objectives = []         # Assuming 'mfe' and 'distance' are the objectives
        
    def __init__(self, task_name, budget_type, budget, seed, workload, **kwargs):
        self.target_structure = self.target_structures[workload]
        self.structure_len = len(self.target_structure)
        self.num_variables = self.structure_len
        self.obj_names = ['mfe', 'distance', 'GCContent', 'success_rate']
        self.obj_funcs = {'mfe': self.cal_mfe, 'distance': self.cal_distance, 'GCContent': self.cal_GCContent, 'success_rate': self.cal_success_rate }

        self.obj_name = kwargs.get('obj_name', 'mfe')
        
        super().__init__(task_name=task_name, budget=budget, budget_type=budget_type, workload=workload, seed=seed)

    def get_configuration_space(self) -> SearchSpace:
        variables = [Categorical(f'aa{i}', ['A', 'U', 'C', 'G']) for i in range(self.structure_len)]
        return SearchSpace(variables)
    
    def get_fidelity_space(self) -> FidelitySpace:
        return FidelitySpace([])
    
    def cal_mfe(self, sequence):
        fold_compound = RNA.fold_compound(sequence)
        structure, mfe = fold_compound.mfe()
        return mfe
    
    def cal_distance(self, sequence):
        distance = self.str_distance(sequence, self.target_structure)
        return distance
    
    def cal_GCContent(self, sequence):
        return self.paired_gc_content(sequence, self.target_structure)
    
    def cal_success_rate(self, sequence):
        return 0
        
    
    @staticmethod
    def str_distance(s1: str, s2: str) -> int:
        """Calculate the Hamming distance between two strings."""
        return sum(el1 != el2 for el1, el2 in zip(s1, s2))
    
    @staticmethod
    def paired_gc_content(sequence, structure):
        """Compute the gc content of paired region."""
        gc_paired_count = 0
        total_paired_count = 0

        stack = []
        for base, symbol in zip(sequence, structure):
            if symbol == '(':
                stack.append(base)
            elif symbol == ')' and stack:
                left_base = stack.pop()
                if {left_base, base} == {'G', 'C'}:
                    gc_paired_count += 1
                total_paired_count += 1

        return gc_paired_count / total_paired_count if total_paired_count else 0.0

    def objective_function(self, configuration: dict, fidelity = None, seed = None, **kwargs):
        """Compute the objective values based on configuration."""
        # Generate RNA sequence by ordered concatenation of configuration values
        sequence = ''.join(configuration[f'aa{i}'] for i in range(self.num_variables))
        
        # Use RNA folding tool to calculate structure and minimum free energy (MFE)
        if self.obj_name in self.obj_names:
            obj_value = self.obj_funcs[self.obj_name](sequence)
        else:
            raise ValueError(f"Invalid objective function: {self.obj_name}. Must be one of {self.obj_names}")
        
        return {
            self.obj_name  : obj_value,
        }
    
    def get_objectives(self) -> dict:
        """Define objectives for optimization: minimizing distance and MFE."""
        return {
            'mfe': 'minimize',       # Objective: minimize free energy
            'distance': 'minimize',   # Objective: minimize structural distance to target
            'GCContent': 'maximize',
            'success_rate': 'maximize'
        }

    def get_problem_type(self) -> str:
        return self.problem_type
    
if __name__ == "__main__":
    target_structures, workloads = get_structures()
    print(f"Total structures loaded: {len(target_structures)}")