import os
import traceback
import argparse
from services import Services
from transopt.analysis.pipeline import analysis, comparison
from transopt.analysis import *


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def set_task(services, args):
    task_info = [{
        "name": args.task_name,
        "num_vars": args.num_vars,
        "num_objs": args.num_objs,
        "fidelity": args.fidelity,
        "workloads": args.workloads,
        "budget_type": args.budget_type,
        "budget": args.budget,
    }]
    services.receive_tasks(task_info)


def set_optimizer(services, args):
    optimizer_info = {
        "SpaceRefiner": args.space_refiner,
        "SpaceRefinerParameters": args.space_refiner_parameters,
        "SpaceRefinerDataSelector": args.space_refiner_data_selector,
        "SpaceRefinerDataSelectorParameters": args.space_refiner_data_selector_parameters,
        "Sampler": args.sampler,
        "SamplerInitNum": args.sampler_init_num,
        "SamplerParameters": args.sampler_parameters,
        "SamplerDataSelector": args.sampler_data_selector,
        "SamplerDataSelectorParameters": args.sampler_data_selector_parameters,
        "Pretrain": args.pre_train,
        "PretrainParameters": args.pre_train_parameters,
        "PretrainDataSelector": args.pre_train_data_selector,
        "PretrainDataSelectorParameters": args.pre_train_data_selector_parameters,
        "Model": args.model,
        "ModelParameters": args.model_parameters,
        "ModelDataSelector": args.model_data_selector,
        "ModelDataSelectorParameters": args.model_data_selector_parameters,
        "ACF": args.acquisition_function,
        "ACFParameters": args.acquisition_function_parameters,
        "ACFDataSelector": args.acquisition_function_data_selector,
        "ACFDataSelectorParameters": args.acquisition_function_data_selector_parameters,
        "Normalizer": args.normalizer,
        "NormalizerParameters": args.normalizer_parameters,
        "NormalizerDataSelector": args.normalizer_data_selector,
        "NormalizerDataSelectorParameters": args.normalizer_data_selector_parameters,
    }
    services.receive_optimizer(optimizer_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("-e", "--experiment_name", type=str, default="exp_2")
    parser.add_argument("-ed", "--experiment_description", type=str, default="")
    # Task
    parser.add_argument("-n", "--task_name", type=str, default="MPB")
    parser.add_argument("-v", "--num_vars", type=int, default=1)
    parser.add_argument("-n", "--task_name", type=str, default="MPB")
    parser.add_argument("-v", "--num_vars", type=int, default=1)
    parser.add_argument("-o", "--num_objs", type=int, default=1)
    parser.add_argument("-f", "--fidelity", type=str, default="")
    parser.add_argument("-w", "--workloads", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50")
    parser.add_argument("-bt", "--budget_type", type=str, default="Num_FEs")
    parser.add_argument("-b", "--budget", type=int, default=22)
    parser.add_argument("-b", "--budget", type=int, default=22)
    # Optimizer
    parser.add_argument("-sr", "--space_refiner", type=str, default="None")
    parser.add_argument("-srp", "--space_refiner_parameters", type=str, default="")
    parser.add_argument("-srd", "--space_refiner_data_selector", type=str, default="None")
    parser.add_argument("-srdp", "--space_refiner_data_selector_parameters", type=str, default="")
    parser.add_argument("-sp", "--sampler", type=str, default="random")
    parser.add_argument("-spi", "--sampler_init_num", type=int, default=11)
    parser.add_argument("-spi", "--sampler_init_num", type=int, default=11)
    parser.add_argument("-spp", "--sampler_parameters", type=str, default="")
    parser.add_argument("-spd", "--sampler_data_selector", type=str, default="None")
    parser.add_argument("-spdp", "--sampler_data_selector_parameters", type=str, default="")
    parser.add_argument("-pt", "--pre_train", type=str, default="None")
    parser.add_argument("-ptp", "--pre_train_parameters", type=str, default="")
    parser.add_argument("-ptd", "--pre_train_data_selector", type=str, default="None")
    parser.add_argument("-ptdp", "--pre_train_data_selector_parameters", type=str, default="")
    parser.add_argument("-m", "--model", type=str, default="RGPE")
    parser.add_argument("-m", "--model", type=str, default="RGPE")
    parser.add_argument("-mp", "--model_parameters", type=str, default="")
    parser.add_argument("-md", "--model_data_selector", type=str, default="None")
    parser.add_argument("-mdp", "--model_data_selector_parameters", type=str, default="")
    parser.add_argument("-acf", "--acquisition_function", type=str, default="EI")
    parser.add_argument("-acfp", "--acquisition_function_parameters", type=str, default="")
    parser.add_argument("-acfd", "--acquisition_function_data_selector", type=str, default="None")
    parser.add_argument("-acfdp", "--acquisition_function_data_selector_parameters", type=str, default="")
    parser.add_argument("-norm", "--normalizer", type=str, default="Standard")
    parser.add_argument("-normp", "--normalizer_parameters", type=str, default="")
    parser.add_argument("-normd", "--normalizer_data_selector", type=str, default="None")
    parser.add_argument("-normdp", "--normalizer_data_selector_parameters", type=str, default="")

    # Seed
    parser.add_argument("-s", "--seeds", type=int, default=1)
    # parser.add_argument("-s", "--seeds", type=str, default="5")


    args = parser.parse_args()
    services = Services(None, None, None)
    # services._initialize_modules()

    # set_task(services, args)
    # set_optimizer(services, args)
    # try:
    #     services._run_optimize_process(seed = args.seeds)
    # except Exception as e:
    #     traceback.print_exc()
        
    # os.makedirs('Results', exist_ok=True)
    # datasets = []
    # experiment_name = args.experiment_name

    # with open('Results/datasets.txt', 'a') as f:
    #     f.write(f"Experiment: {experiment_name}\n")
    #     for pid, info in services.process_info.items():
    #         dataset_list = info['dataset_name']
    #         datasets += dataset_list
    #         [f.write(f"{dataset}\n") for dataset in dataset_list]
    #     f.write("-----\n")
    

    comparison_experiment_names = ['exp_1']
    comparison_datasets = {}
    analysis_datasets = {}
    with open('Results/datasets.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("Experiment:"):
                experiment_name = lines[i].strip().split(":")[1].strip()
                dataset_list = []
                i += 1
                while i < len(lines) and not lines[i].startswith("-----"):
                    dataset_list.append(lines[i].strip())
                    i += 1
                comparison_datasets[experiment_name] = dataset_list
                analysis_datasets[experiment_name] = dataset_list
    comparison('Results', comparison_datasets, services.data_manager, args)
    
    analysis('Results', analysis_datasets, services.data_manager, args)