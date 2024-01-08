import sys
from pathlib import Path

current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))

import json

import numpy as np
import pandas as pd

from transopt.utils.pareto import calc_hypervolume, find_pareto_front
from transopt.utils.plot import plot3D

results_path = package_path / "experiment_results"
gcc_results = results_path / "gcc_archive_new"

algorithm_list = ["ParEGO", "SMSEGO", "MoeadEGO", "CauMO"]
objectives = ["execution_time", "file_size", "compilation_time"]
seed_list = [65535, 65536, 65537, 65538, 65539]


def load_and_prepare_data(file_path, objectives):
    """
    Loads JSON data and prepares a DataFrame.
    """
    # print(f"Loading data from {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        data = data.get("1", {})

    input_vectors = data["input_vector"]
    output_vectors = data["output_value"]

    df_input = pd.DataFrame(input_vectors)

    df_output = pd.DataFrame(output_vectors)[objectives]
    df_combined = pd.concat([df_input, df_output], axis=1)
    # print(f"Loaded {len(df_combined)} data points")

    df_combined = df_combined.drop_duplicates(subset=df_input.columns.tolist())

    for obj in objectives:
        df_combined = df_combined[df_combined[obj] != 1e10]

    # print(f"Loaded {len(df_combined)} data points, removed {len(df_input) - len(df_combined)} duplicates")
    # print()
    return df_combined


def collect_all_data(workload):
    all_data = []
    for algorithm in algorithm_list:
        for seed in seed_list:
            result_file = (
                gcc_results / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"
            )
            df = load_and_prepare_data(result_file, objectives)
            all_data.append(df[objectives].values)
    all_data = np.vstack(all_data)
    global_mean = all_data.mean(axis=0)
    global_std = all_data.std(axis=0)
    return all_data, global_mean, global_std


def calculate_mean_hypervolume(
    algorithm, workload, global_stats1, global_stats2, normalization_type="min-max"
):
    """
    Calculate mean hypervolume for a given algorithm across all seeds.

    Parameters:
    global_stats1: Global mean or min of all objectives (depending on normalization_type)
    global_stats2: Global std or max of all objectives (depending on normalization_type)
    normalization_type: 'min-max' or 'mean' for different types of normalization
    """
    hypervolume_list = []
    for seed in seed_list:
        result_file = gcc_results / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"
        df = load_and_prepare_data(result_file, objectives)

        if normalization_type == "mean":
            # Apply mean normalization
            normalized_df = (df[objectives] - global_stats1) / global_stats2
        elif normalization_type == "min-max":
            # Apply min-max normalization
            normalized_df = (df[objectives] - global_stats1) / (
                global_stats2 - global_stats1
            )
        else:
            raise ValueError(
                "Unsupported normalization type. Choose 'mean' or 'min-max'."
            )

        pareto_front = find_pareto_front(normalized_df.values)
        hypervolume = calc_hypervolume(pareto_front, np.ones(len(objectives)))
        # print(f"{algorithm} {seed} {hypervolume}")
        hypervolume_list.append(hypervolume)

    return np.mean(hypervolume_list)


def load_workloads():
    file_path = package_path / "demo" / "comparison" / "features_by_workload_gcc.json"
    with open(file_path, "r") as f:
        return json.load(f).keys()


if __name__ == "__main__":
    workloads = load_workloads()

    workloads = list(workloads)
    workloads.sort()

    for workload in workloads:
        print(workload)
        all_data, global_mean, global_std = collect_all_data(workload)
        global_max = all_data.max(axis=0)
        global_min = all_data.min(axis=0)

        hv_list = []
        for algorithm in algorithm_list:
            mean_hypervolume = calculate_mean_hypervolume(
                algorithm,
                workload,
                global_min,
                global_max,
                normalization_type="min-max",
            )
            # mean_hypervolume = calculate_mean_hypervolume(algorithm, workload, global_mean, global_std, normalization_type='mean')
            # print(f"{algorithm} {mean_hypervolume}")
            hv_list.append((algorithm, mean_hypervolume))

        # Sort by hypervolume
        hv_list.sort(key=lambda x: x[1], reverse=True)

        print(hv_list)
        print()
