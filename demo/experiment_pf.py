import argparse
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
sys.path.insert(0, package_dir)

from transopt.Benchmark import construct_test_suits
from transopt.KnowledgeBase.kb_builder import construct_knowledgebase
from transopt.KnowledgeBase.TaskDataHandler import OptTaskDataHandler
from transopt.Optimizer.ConstructOptimizer import get_optimizer

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def run_experiments(tasks, args):
    logger = logging.getLogger(__name__)
    kb = construct_knowledgebase(args)
    testsuits = construct_test_suits(tasks, args.seed)
    optimizer = get_optimizer(args)
    data_handler = OptTaskDataHandler(kb, args)
    optimizer.optimize(testsuits, data_handler)


if __name__ == "__main__":
    folders = {
        "exp_path": f"{package_dir}/../experiment_results",
    }

    tasks = {
        "AckleySphere": {
            "budget": 1000,
            "workloads": [1, 2, 3],
            "params": {"input_dim": 2},
        },
    }

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-im", "--init_method", type=str, default="random")
    parser.add_argument("-in", "--init_number", type=int, default=7)
    parser.add_argument("-p", "--exp_path", type=str, default=folders["exp_path"])
    parser.add_argument("-n", "--exp_name", type=str, default="test_pf")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-m", "--optimizer", type=str, default="ParEGO")
    parser.add_argument("-v", "--verbose", type=bool, default=True)
    parser.add_argument("-norm", "--normalize", type=str, default="norm")
    parser.add_argument("-sm", "--save_mode", type=int, default=1)
    parser.add_argument("-lm", "--load_mode", type=bool, default=False)
    parser.add_argument("-ac", "--acquisition_func", type=str, default="LCB")
    args = parser.parse_args()

    run_experiments(tasks, args)
