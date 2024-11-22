import json
import zipfile

import numpy as np
import requests
import xgboost as xgb

from pathlib import Path
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.utils.path import get_library_path

# Prepare data (download and extract surrogates) at module import time
data_dir = get_library_path() / "hpob_data"
surrogates_zip_path = data_dir / "saved-surrogates.zip"
surrogates_dir = data_dir / "saved-surrogates"
summary_path = surrogates_dir / "summary-stats.json"

if not surrogates_dir.exists():
    url = "https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip"
    data_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(surrogates_zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    with zipfile.ZipFile(surrogates_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

if summary_path.exists():
    with open(summary_path, "r") as f:
        summary_stats = json.load(f)
        workloads = [entry.replace("surrogate-", "") for entry in summary_stats.keys()]
else:
    raise FileNotFoundError("Summary stats file not found in the surrogate directory.")


@problem_registry.register("HPOB")
class HPOB(NonTabularProblem):
    problem_type = "HPOB"
    fidelity = None
    workloads = workloads
    num_variables = 0
    num_objectives = 1

    def __init__(self, task_name, budget_type, budget, seed, workload, **kwargs):
        self.workload = workload
        self.surrogate_model = self.load_surrogate(workload)
        self.num_variables = self.get_input_dimension(workload)
        super().__init__(task_name=task_name, budget=budget, budget_type=budget_type, workload=workload, seed=seed)

    def load_surrogate(self, workload):
        surrogate_name = f"surrogate-{workload}.json"
        surrogate_path = surrogates_dir / surrogate_name
        if not surrogate_path.exists():
            raise FileNotFoundError(f"Surrogate model not found for the given workload: {surrogate_name}")
        
        bst = xgb.Booster()
        bst.load_model(str(surrogate_path))
        return bst
    
    def get_input_dimension(self, workload):
        surrogate_name = f"surrogate-{workload}.json"
        surrogate_path = surrogates_dir / surrogate_name
        if not surrogate_path.exists():
            raise FileNotFoundError(f"Surrogate model not found for the given workload: {surrogate_name}")
        
        with open(surrogate_path, "r") as f:
            model_info = json.load(f)
            learner_model_param = model_info.get("learner", {}).get("learner_model_param", {})
            dimension = int(learner_model_param.get("num_feature", 0))
            
        if dimension > 0:
            return dimension
        else:
            raise ValueError(f"Input dimension for workload {workload} could not be determined from the model dump.")
        
    def get_configuration_space(self) -> SearchSpace:
        variables = [Continuous(f'x{i}', [0.0, 1.0]) for i in range(self.num_variables)]
        return SearchSpace(variables)

    def get_fidelity_space(self) -> FidelitySpace:
        return FidelitySpace([])

    def evaluate_continuous(self, config_vector):
        dmatrix = xgb.DMatrix(np.array(config_vector).reshape(1, -1))
        performance = self.surrogate_model.predict(dmatrix)[0]
        return performance

    def objective_function(self, configuration: dict, fidelity=None, seed=None, **kwargs):
        config_vector = [configuration.get(f'x{i}', 0.0) for i in range(self.num_variables)]
        obj_value = self.evaluate_continuous(config_vector)
        return {
            'objective': obj_value
        }

    def get_objectives(self) -> dict:
        return {
            'objective': 'maximize'
        }

    def get_problem_type(self) -> str:
        return self.problem_type


# Example usage
if __name__ == "__main__":
    workload = "5624-43"
    hpob_problem = HPOB(task_name="HPOB-Test", budget_type="time", budget=1000, seed=42, workload=workload)
    input_dim = hpob_problem.num_variables
    config_vector = [0.5] * input_dim  # Use input_dim to create a matching config vector length
    predicted_performance = hpob_problem.evaluate_continuous(config_vector)
    print("Predicted performance of configuration (using surrogate model):", predicted_performance)
