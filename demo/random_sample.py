import logging
import os
import argparse
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
sys.path.insert(0, package_dir)

from csstuning.compiler.compiler_benchmark import CompilerBenchmarkBase
from transopt.Benchmark import construct_test_suits
from transopt.Optimizer.ConstructOptimizer import get_optimizer
from transopt.KnowledgeBase.kb_builder import construct_knowledgebase
from transopt.KnowledgeBase.TaskDataHandler import OptTaskDataHandler


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def run_experiments(tasks, args):
    kb = construct_knowledgebase(args)
    testsuits = construct_test_suits(tasks, args.seed)
    optimizer = get_optimizer(args)
    data_handler = OptTaskDataHandler(kb, args)
    optimizer.optimize(testsuits, data_handler)


def split_into_segments(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "split_index", type=int, help="Index for splitting the workload segments"
    )
    args = parser.parse_args()
    split_index = args.split_index


    samples_num = 5
    available_workloads = CompilerBenchmarkBase.AVAILABLE_WORKLOADS
    split_workloads = split_into_segments(available_workloads, 10)[0]

    if split_index >= len(split_workloads):
        raise IndexError("split index out of range")

    workloads = split_workloads[split_index]

    tasks = {
        "GCC": {"budget": samples_num, "workloads": workloads},
        "LLVM": {"budget": samples_num, "workloads": workloads},
    }

    args = argparse.Namespace(
        seed=0,
        optimizer="ParEGO",
        init_number=samples_num,
        init_method="random",
        exp_path=f"{package_dir}/../experiment_results",
        exp_name="sampling_compiler",
        verbose=True,
        normalize="norm",
        source_num=2,
        selector="None",
        save_mode=1,
        load_mode=False,
        acquisition_func="LCB",
    )

    run_experiments(tasks, args)
