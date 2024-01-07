import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
package_dir = current_dir.parent.parent
sys.path.insert(0, str(package_dir))

import argparse
import json
import os

import numpy as np
from csstuning.compiler.compiler_benchmark import CompilerBenchmarkBase

from transopt.Benchmark import construct_test_suits
from transopt.KnowledgeBase.kb_builder import construct_knowledgebase
from transopt.KnowledgeBase.TaskDataHandler import OptTaskDataHandler
from transopt.Optimizer.ConstructOptimizer import get_optimizer

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def execute_tasks(tasks, args):
    kb = construct_knowledgebase(args)
    testsuits = construct_test_suits(tasks, args.seed)
    optimizer = get_optimizer(args)
    data_handler = OptTaskDataHandler(kb, args)
    optimizer.optimize(testsuits, data_handler)


def split_into_segments(lst, n):
    lst = list(lst)
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def get_workloads(workloads, split_index, total_splits=10):
    segments = split_into_segments(workloads, total_splits)
    if split_index >= len(segments):
        raise IndexError("split index out of range")

    return segments[split_index]


def load_features():
    file_path = package_dir / "demo" / "comparison" / "features_by_workload_gcc.json"
    with open(file_path, "r") as f:
        return json.load(f)


def configure_experiment(workload, features, seed, optimizer_name, exp_path, budget=20, init_number=10):
    exp_name = f"gcc_{workload}"
    args = argparse.Namespace(
        seed=seed,
        optimizer=optimizer_name,
        budget=budget,
        init_number=init_number,
        pop_size=init_number,
        init_method="random",
        exp_path=exp_path,
        exp_name=exp_name,
        verbose=True,
        normalize="norm",
        acquisition_func="LCB",
    )
    tasks = {
        "GCC": {
            "budget": budget,
            "workloads": [workload],
            "knobs": features[workload]["top"],
        },
    }
    return tasks, args

def main(optimizers = [], repeat=5, budget=500, init_number=21):
    features = load_features()

    parser = argparse.ArgumentParser(description="Run optimization experiments")
    parser.add_argument("--split_index", type=int, default=0,
                        help="Index for splitting the workload segments")
    args = parser.parse_args()

    workloads = get_workloads(features.keys(), args.split_index)

    exp_path = package_dir / "experiment_results"

    for optimizer_name in optimizers:
        for workload in workloads:
            for i in range(repeat):
                tasks, exp_args = configure_experiment(
                    workload,
                    features,
                    65535 + i,
                    optimizer_name,
                    exp_path,
                    budget,
                    init_number,
                )
                execute_tasks(tasks, exp_args)


def main_debug(repeat=1, budget=20, init_number=10):
    features = load_features()

    parser = argparse.ArgumentParser(description="Run optimization experiments")
    parser.add_argument("--split_index", type=int, default=9,
                        help="Index for splitting the workload segments")
    args = parser.parse_args()

    workloads = get_workloads(features.keys(), args.split_index)[:1]

    workloads = ["cbench-consumer-jpeg-d"]
    exp_path = package_dir / "experiment_results"

    for optimizer_name in ["MoeadEGO"]:
        for workload in workloads:
            for i in range(repeat):
                tasks, exp_args = configure_experiment(
                    workload,
                    features,
                    65535 + i,
                    optimizer_name,
                    exp_path,
                    budget,
                    init_number,
                )
                execute_tasks(tasks, exp_args)


if __name__ == "__main__":
    debug = True
    debug = False
    if debug:
        main_debug(repeat=5, budget=500, init_number=10)
    else:
        main(["CauMO"], repeat=5, budget=500, init_number=21)
