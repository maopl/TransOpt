"""
Example with XGBoost (local)
============================
This example executes the xgboost benchmark locally with random configurations on the CC18 openml tasks.

To run this example please install the necessary dependencies via:
``pip3 install .[xgboost_example]``
"""

import argparse
from time import time

from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark as Benchmark
from hpobench.util.openml_data_manager import get_openmlcc18_taskids
from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark as TabBenchmarkContainer
from hpobench.benchmarks.ml.svm_benchmark_old import svm

def run_experiment_bench(on_travis: bool = False):
    task_ids = get_openmlcc18_taskids()
    # for task_no, task_id in enumerate(task_ids):
    task_id = 167149
    # if on_travis and task_no == 5:
    #     break

    print(f'# ################### TASK {167149 + 1} of {len(task_ids)}: Task-Id: {task_id} ################### #')
    # if task_id == 167204 or task_id==167150:
    #     continue  # due to memory limits

    b = Benchmark(task_id=task_id)
    cs = b.get_configuration_space()
    print(cs)
    start = time()
    num_configs = 2
    for i in range(num_configs):
        configuration = cs.sample_configuration()
        print(configuration)
        for i in range(5):
        # for n_estimator in [8, 64]:
        #     for subsample in [0.4, 1]:
        # fidelity = {'n_estimators': n_estimator, 'dataset_fraction': subsample}
            result_dict = b.objective_function(configuration.get_dictionary())
            valid_loss = result_dict['function_value']
            train_loss = result_dict['info']['train_loss']

            result_dict = b.objective_function_test(configuration)
            test_loss = result_dict['function_value']
            #
            print(f'[{i+1}|{num_configs}] repeat_id{i}  - Test {test_loss:.4f} '
                  f'- Valid {valid_loss:.4f} - Train {train_loss:.4f}')

    print(f'Done, took totally {time()-start:.2f}')



def run_experiment_container(on_travis=False):

    benchmark = TabBenchmarkContainer(container_name='tabular_benchmarks',
                                      container_source='library://phmueller/automl',
                                      rng=1)

    cs = benchmark.get_configuration_space(seed=1)
    config = cs.sample_configuration()
    print(config)

    # You can pass the configuration either as a dictionary or a ConfigSpace.configuration
    result_dict_1 = benchmark.objective_function(configuration=config.get_dictionary())
    result_dict_2 = benchmark.objective_function(configuration=config)
    print(result_dict_1, result_dict_2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='HPOBench CC Datasets', description='HPOBench on the CC18 data sets.',
                                     usage='%(prog)s --array_id <task_id>')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \'travis\'. This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment_bench(on_travis=args.on_travis)