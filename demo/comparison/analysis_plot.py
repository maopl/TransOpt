import sys
from pathlib import Path

current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))

import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from transopt.utils.pareto import calc_hypervolume, find_pareto_front
from transopt.utils.plot import plot3D

results_path = package_path / "experiment_results"
gcc_results_path = results_path / "gcc_archive"
gcc_samples_path = results_path / "gcc_samples"

algorithm_list = ["ParEGO", "SMSEGO", "MoeadEGO", "CauMO"]
# algorithm_list = ["SMSEGO"]
objectives = ["execution_time", "file_size", "compilation_time"]
seed_list = [65535, 65536, 65537, 65538, 65539]


def load_and_prepare_data(file_path):
    """
    Loads JSON data and prepares a DataFrame.
    """
    # print(f"Loading data from {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        if "1" in data:
            data = data["1"]

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
            result_file = gcc_results_path / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"
            df = load_and_prepare_data(result_file)
            all_data.append(df[objectives].values)
    all_data = np.vstack(all_data)
    global_mean = all_data.mean(axis=0)
    global_std = all_data.std(axis=0)
    return all_data, global_mean, global_std


def dynamic_plot(workload, algorithm, seed):
    """
    Dynamically plot the three objectives for a given workload and algorithm for a specific seed.
    """
    # Collect all data to understand the range
    all_data, global_mean, global_std = collect_all_data(workload)
    global_min = np.min(all_data, axis=0)
    global_max = np.max(all_data, axis=0)
   
    # Load data for the specific seed
    result_file = gcc_results_path / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"
    df = load_and_prepare_data(result_file)
    
    # Normalize data (Min-Max normalization)
    df_normalized = (df - global_min) / (global_max - global_min)
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Dynamic Plot for {workload} - {algorithm} - Seed {seed}")
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2])

    # Initialize two scatter plots: one for all previous points, one for the new point
    previous_points = ax.scatter([], [], [], c='b', marker='o')  # all previous points in blue
    current_point = ax.scatter([], [], [], c='r', marker='o')  # current point in red
    
    def init():
        previous_points._offsets3d = ([], [], [])
        current_point._offsets3d = ([], [], [])
        return previous_points, current_point

    def update(frame):
        # Add all previous points up to the current frame
        previous_points._offsets3d = (df_normalized.iloc[:frame][objectives[0]].values,
                                      df_normalized.iloc[:frame][objectives[1]].values,
                                      df_normalized.iloc[:frame][objectives[2]].values)

        # Add the current point (latest one in the sequence)
        current_point._offsets3d = (df_normalized.iloc[frame:frame+1][objectives[0]].values,
                                    df_normalized.iloc[frame:frame+1][objectives[1]].values,
                                    df_normalized.iloc[frame:frame+1][objectives[2]].values)
        return previous_points, current_point
    
    frames = len(df)
    ani = FuncAnimation(fig, update, frames=frames, blit=False, repeat=False)
    
    # Save the plot to a file
    gif_path = package_path / "demo" / "comparison" / "gifs" / f"{algorithm}_{workload}_{seed}.gif"
    ani.save(gif_path, writer='imagemagick')
    plt.close(fig)  # Close the plot to free memory


def save_individual_frames(workload, algorithm, seed):
    """
    Save each frame of the three objectives as a separate plot for a given workload, algorithm, and seed.
    """
    # Load data for the specific seed
    result_file = gcc_results_path / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"
    df = load_and_prepare_data(result_file)

    # Ensure the directory for saving frames exists
    frames_dir = package_path / "demo" / "comparison" / "frames" / f"{algorithm}_{workload}_{seed}"
    os.makedirs(frames_dir, exist_ok=True)
    
    for idx in range(len(df)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Add data points from the DataFrame row by row
        x, y, z = df.iloc[idx][objectives[0]], df.iloc[idx][objectives[1]], df.iloc[idx][objectives[2]]
        
        # Plot and customize as needed
        ax.scatter(x, y, z, color='r')
        ax.set_title(f"Frame {idx} for {workload} - {algorithm} - Seed {seed}")
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        
        # Save the plot as a file
        frame_file = frames_dir / f"frame_{idx:04d}.png"
        plt.savefig(frame_file)
        plt.close(fig)  # Close the plot to free memory
        

def load_workloads():
    file_path = package_path / "demo" / "comparison" / "features_by_workload_gcc.json"
    with open(file_path, "r") as f:
        return json.load(f).keys()

def plot_pareto_front(workload):
    df = load_and_prepare_data(gcc_samples_path / f"GCC_{workload}.json")
    df_normalized = (df - df.min()) / (df.max() - df.min())
    pareto_front, pareto_indices = find_pareto_front(df_normalized[objectives].values, return_index=True)
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Pareto Front for {workload}")
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2])
    
    # Scatter plot for Pareto front
    pareto_points = df_normalized.iloc[pareto_indices][objectives]
    # pareto_points = df_normalized[objectives]
    
    # Convert Series to NumPy array before plotting
    x_values = pareto_points[objectives[0]].values
    y_values = pareto_points[objectives[1]].values
    z_values = pareto_points[objectives[2]].values

    ax.scatter(x_values, y_values, z_values, c='b', marker='o')

    # ax.plot(x_values, y_values, z_values, color='b')
    
    # Save the plot as a file
    file_path = package_path / "demo" / "comparison" / "pngs" / f"pareto_front_{workload}.png"
    plt.savefig(file_path)
    plt.close(fig)  # Close the plot to free memory
    
def plot_all(workload):
    df = load_and_prepare_data(gcc_samples_path / f"GCC_{workload}.json")
    df_normalized = (df - df.min()) / (df.max() - df.min())
    pareto_front, pareto_indices = find_pareto_front(df_normalized[objectives].values, return_index=True)
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Pareto Front for {workload}")
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2])
    
    # Scatter plot for Pareto front
    # pareto_points = df_normalized.iloc[pareto_indices][objectives]
    pareto_points = df_normalized[objectives]
    
    # Convert Series to NumPy array before plotting
    x_values = pareto_points[objectives[0]].values
    y_values = pareto_points[objectives[1]].values
    z_values = pareto_points[objectives[2]].values

    ax.scatter(x_values, y_values, z_values, c='b', marker='o')

    # ax.plot(x_values, y_values, z_values, color='b')
    
    # Save the plot as a file
    file_path = package_path / "demo" / "comparison" / "pngs" / f"all_{workload}.png"
    plt.savefig(file_path)
    plt.close(fig)  # Close the plot to free memory

if __name__ == "__main__":
    workloads = load_workloads()
    # jpeg-d miss moead, sha miss moead seed 65539, delete them
    workloads -= {"cbench-consumer-jpeg-d", "cbench-security-sha"} 

    workloads = [
        "cbench-security-pgp", 
        "polybench-cholesky",
        "cbench-consumer-tiff2rgba",
        "cbench-network-patricia",
        "cbench-automotive-susan-e",
        "polybench-symm",
        "cbench-consumer-mad",
        "polybench-lu"
    ]
    
    seed = 65535  # Example seed
    for workload in workloads:
        plot_all(workload)
        plot_pareto_front(workload)
    
    for algorithm in ["CauMO"]:


        pass 
        # dynamic_plot(workload, algorithm, objectives, seed)
        # save_individual_frames(workload, algorithm, objectives, seed)