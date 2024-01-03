import sys
from pathlib import Path

current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))

import json
import pandas as pd

from transopt.utils.pareto import find_pareto_front
from transopt.utils.plot import plot3D

results_path = package_path / "experiment_results"


def load_and_prepare_data(file_path, objectives):
    """
    Loads JSON data and prepares a DataFrame.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
        data = data.get("1", {})

    input_vectors = data["input_vector"]
    output_vectors = data["output_value"]

    df_input = pd.DataFrame(input_vectors)

    df_output = pd.DataFrame(output_vectors)[objectives]
    df_combined = pd.concat([df_input, df_output], axis=1)
    print(f"Loaded {len(df_combined)} data points")

    df_combined = df_combined.drop_duplicates(subset=df_input.columns.tolist())
    print(f"Removed {len(df_combined) - len(df_input)} duplicates")

    for obj in objectives:
        df_combined = df_combined[df_combined[obj] != 1e10]
    print(f"Loaded {len(df_combined)} data points after removing extreme values")
    return df_combined


if __name__ == "__main__":
    gcc_results = results_path / "gcc_archive"

    workload = "cbench-automotive-qsort1"
    algorithm = "ParEGO"
    seed = 65535
    result_file = gcc_results / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"

    objectives = ["execution_time", "file_size", "compilation_time"]
    df = load_and_prepare_data(result_file, objectives)
    
    # get pareto front and indices
    pareto_front, pareto_indices = find_pareto_front(df[objectives].values, return_index=True)
    print(f"Found {len(pareto_front)} pareto optimal points")
    for i in pareto_indices:
        print(df.iloc[i][objectives])
        print()

    # Plot pareto front
    plot3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], show=True)

