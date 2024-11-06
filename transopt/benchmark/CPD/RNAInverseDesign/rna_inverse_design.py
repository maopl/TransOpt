import os

import numpy as np
import pandas as pd
import RNA

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *


@problem_registry.register("RNAInverseDesign")
class RNAInverseDesign(NonTabularProblem):
    problem_type = "RNAInverseDesign"
    fidelity = None
    _target_structures = None  # Internal storage for lazy-loaded target structures
    _workloads = None          # Internal storage for lazy-loaded workloads
    num_variables = 0          # Based on workload
    num_objectives = 2         # Assuming 'mfe' and 'distance' are the objectives

    @classmethod
    def get_structures(cls):
        """Load RNA target structures from a pre-defined file if not already loaded."""
        if cls._target_structures is None:
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, 'inverse_rna_folding_benchmark_dotbracket.pkl.gz')
            df = pd.read_pickle(file_path)
            cls._target_structures = df['dotbracket'].tolist()
            cls._workloads = list(range(len(cls._target_structures)))
        return cls._target_structures

    @property
    def target_structures(self):
        """Property to access target structures, triggering lazy initialization."""
        return self.get_structures()

    @property
    def workloads(self):
        """Property to access workloads, triggering lazy initialization."""
        self.get_structures()  # Ensures _workloads is set
        return self._workloads
        
    def __init__(self, task_name, budget_type, budget, seed, workload, **kwargs):
        self.target_structure = self.target_structures[workload]
        self.structure_len = len(self.target_structure)
        self.num_variables = self.structure_len
        
        super().__init__(task_name=task_name, budget=budget, budget_type=budget_type, workload=workload, seed=seed)

    def get_configuration_space(self) -> SearchSpace:
        variables = [Categorical(f'aa{i}', ['A', 'U', 'C', 'G']) for i in range(self.structure_len)]
        return SearchSpace(variables)
    
    def get_fidelity_space(self) -> FidelitySpace:
        return FidelitySpace([])
    
    @staticmethod
    def str_distance(s1: str, s2: str) -> int:
        """Calculate the Hamming distance between two strings."""
        return sum(el1 != el2 for el1, el2 in zip(s1, s2))

    def objective_function(self, configuration: dict, fidelity = None, seed = None, **kwargs):
        """Compute the objective values based on configuration."""
        # Generate RNA sequence by ordered concatenation of configuration values
        sequence = ''.join(configuration[f'aa{i}'] for i in range(self.num_variables))
        
        # Use RNA folding tool to calculate structure and minimum free energy (MFE)
        fold_compound = RNA.fold_compound(sequence)
        structure, mfe = fold_compound.mfe()
        
        # Calculate distance from target structure
        distance = self.str_distance(structure, self.target_structure)
        
        return {
            'mfe': mfe,
            'distance': distance
        }
    
    def get_objectives(self) -> dict:
        """Define objectives for optimization: minimizing distance and MFE."""
        return {
            'mfe': 'minimize',       # Objective: minimize free energy
            'distance': 'minimize'   # Objective: minimize structural distance to target
        }

    def get_problem_type(self) -> str:
        return self.problem_type
    
if __name__ == "__main__":
    benchmark = RNAInverseDesign("test", "iterations", 100, 1, 1)
    print(len(benchmark.target_structures))
    print(benchmark.target_structures[0])