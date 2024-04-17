import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
package_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(package_dir))

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from csstuning.compiler.compiler_benchmark import GCCBenchmark, LLVMBenchmark

from transopt.Benchmark.BenchBase import NonTabularBenchmark
from transopt.space import DesignSpace
from transopt.utils.Register import benchmark_register



@benchmark_register("GCC")
class GCCTuning(NonTabularBenchmark):
    ERROR_VALUE = 1e10
    
    def __init__(
        self, task_name, budget, seed, task_type="non-tabular", workload=None, knobs=None, **kwargs
    ):
        self.objectives = ["execution_time", "file_size", "compilation_time"]
        self.workload = workload if workload is not None else GCCBenchmark.AVAILABLE_WORKLOADS[0]
        self.benchmark = GCCBenchmark(workload=self.workload)
       
        # Initialize DesignSpace
        self.design_space = DesignSpace()
        all_knobs = self.benchmark.config_space.get_all_details()
        knob_details = [all_knobs[k] for k in (knobs if knobs else all_knobs.keys())]
        params_cfg = self.parse_configuration(knob_details)
        self.design_space.parse(params_cfg)
         
        super().__init__(task_name=task_name, workload=self.workload, seed=seed, task_type=task_type, budget=budget)
        
        np.random.seed(seed)

    def parse_configuration(self, knobs: list) -> dict:
        cfg = []
        for knob in knobs:
            p_cfg = {'name': knob['name']}
            if knob["type"] == "enum":
                p_cfg["type"] = "categorical"
                p_cfg["categories"] = knob["range"]
            elif knob["type"] == "integer":
                p_cfg["type"] = "integer"
                p_cfg["lb"] = knob["range"][0]
                p_cfg["ub"] = knob["range"][1]
            else:
                raise TypeError
            
            p_cfg["default"] = knob["default"]
            cfg.append(p_cfg)
        return cfg 
    
    def objective_function(self, configuration: pd.DataFrame, fidelity, seed, **kwargs) -> (pd.DataFrame, float):
        cfg_dict = configuration.to_dict(orient="records")[0]
        
        try:
            start_time = time.time()
            performance = self.benchmark.run(cfg_dict)
            end_time = time.time()

            results = {
                "avrg_exec_time": float(performance["avrg_exec_time"]),
                "file_size": float(performance["file_size"]),
                "PAPI_TOT_CYC": float(performance["PAPI_TOT_CYC"]),
                "PAPI_TOT_INS": float(performance["PAPI_TOT_INS"]),
                "PAPI_BR_MSP": float(performance["PAPI_BR_MSP"]),
                "PAPI_BR_PRC": float(performance["PAPI_BR_PRC"]),
                "PAPI_BR_CN": float(performance["PAPI_BR_CN"]),
                "PAPI_MEM_WCY": float(performance["PAPI_MEM_WCY"]),
                "maxrss": float(performance["maxrss"]),
                "compilation_time": float(performance["compilation_time"]),
                "execution_time": float(performance["execution_time"]),
            }
        except:
            results = {
                "avrg_exec_time": float(self.ERROR_VALUE),
                "file_size": float(self.ERROR_VALUE),
                "PAPI_TOT_CYC": float(self.ERROR_VALUE),
                "PAPI_TOT_INS": float(self.ERROR_VALUE),
                "PAPI_BR_MSP": float(self.ERROR_VALUE),
                "PAPI_BR_PRC": float(self.ERROR_VALUE),
                "PAPI_BR_CN": float(self.ERROR_VALUE),
                "PAPI_MEM_WCY": float(self.ERROR_VALUE),
                "maxrss": float(self.ERROR_VALUE),
                "compilation_time": float(self.ERROR_VALUE),
                "execution_time": float(self.ERROR_VALUE),
            } 

        
        
        return results, float(end_time - start_time)
            
        

    def get_configuration_space(
        self, seed: Union[int, None] = None
    ):
        cs = CS.ConfigurationSpace(seed=seed or np.random.randint(1, 100000))

        hyperparameters = []
        self.min_value = {}
        for key, value in self.config_knob.items():
            if value["type"] == "enum":
                hyperparameters.append(
                    CS.CategoricalHyperparameter(
                        key, choices=value["range"], default_value=str(value["default"])
                    )
                )
            elif value["type"] == "integer":
                hyperparameters.append(
                    CS.UniformIntegerHyperparameter(
                        key,
                        lower=value["range"][0],
                        upper=value["range"][1],
                        default_value=value["default"],
                    )
                )
            else:
                raise TypeError

        cs.add_hyperparameters(hyperparameters)

        return cs

    def get_fidelity_space(
        self, seed: Union[int, None] = None
    ):
        return CS.ConfigurationSpace(seed=seed or np.random.randint(1, 100000))

    def get_meta_information(self) -> Dict:
        return {"number_objective": len(self.objectives)}


# @benchmark_register("LLVM")
# class LLVMTuning(NonTabularBenchmark):
#     def __init__(
#         self, task_name, budget, seed, task_type="non-tabular", workload=None, knobs=None, **kwargs
#     ):
#         self.objectives = ["execution_time", "file_size", "compilation_time"]

#         self.workload = workload if workload is not None else GCCBenchmark.AVAILABLE_WORKLOADS[0]
#         self.benchmark = LLVMBenchmark(workload=self.workload)
        
#         all_knobs = self.benchmark.config_space.get_all_details()
#         self.config_knob = {k: all_knobs[k] for k in knobs} if knobs else all_knobs

#         super().__init__(task_name=task_name, workload=self.workload, seed=seed, task_type=task_type, budget=budget)
#         np.random.seed(seed)

#     def objective_function(
#         self,
#         configuration: Union[ConfigSpace.Configuration, Dict],
#         fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
#         seed: Union[np.random.RandomState, int, None] = None,
#         **kwargs
#     ) -> Dict:

#         c = {}
#         for k, v in configuration.items():
#             assert k in self.config_knob, f"Invalid configuration key: {k}"
#             knob_type = self.config_knob[k]["type"]
#             if knob_type == "enum":
#                 c[k] = self.config_knob[k]["range"][int(v)]
#             elif knob_type == "integer":
#                 c[k] = int(v)

#         try:
#             start_time = time.time()
#             performance = self.benchmark.run(c)
#             end_time = time.time()

#             results = {
#                 "avrg_exec_time": float(performance["avrg_exec_time"]),
#                 "file_size": float(performance["file_size"]),
#                 "PAPI_TOT_CYC": float(performance["PAPI_TOT_CYC"]),
#                 "PAPI_TOT_INS": float(performance["PAPI_TOT_INS"]),
#                 "PAPI_BR_MSP": float(performance["PAPI_BR_MSP"]),
#                 "PAPI_BR_PRC": float(performance["PAPI_BR_PRC"]),
#                 "PAPI_BR_CN": float(performance["PAPI_BR_CN"]),
#                 "PAPI_MEM_WCY": float(performance["PAPI_MEM_WCY"]),
#                 "maxrss": float(performance["maxrss"]),
#                 "compilation_time": float(performance["compilation_time"]),
#                 "execution_time": float(performance["execution_time"]),
#                 "cost": float(end_time - start_time),
#                 "info": {"fidelity": fidelity},
#             }

#             results.update({
#                 f"function_value_{i + 1}": results[objective]
#                 for i, objective in enumerate(self.objectives)
#             })

#             return results
            
#         except:
#             error_results = {
#                 "avrg_exec_time": float(ERROR_VALUE),
#                 "file_size": float(ERROR_VALUE),
#                 "PAPI_TOT_CYC": float(ERROR_VALUE),
#                 "PAPI_TOT_INS": float(ERROR_VALUE),
#                 "PAPI_BR_MSP": float(ERROR_VALUE),
#                 "PAPI_BR_PRC": float(ERROR_VALUE),
#                 "PAPI_BR_CN": float(ERROR_VALUE),
#                 "PAPI_MEM_WCY": float(ERROR_VALUE),
#                 "maxrss": float(ERROR_VALUE),
#                 "compilation_time": float(ERROR_VALUE),
#                 "execution_time": float(ERROR_VALUE),
#                 "cost": float(end_time - start_time),
#                 "info": {"fidelity": fidelity},
#             }

#             error_results.update({
#                 f"function_value_{i + 1}": error_results[objective]
#                 for i, objective in enumerate(self.objectives)
#             })
#             return error_results
    
#     def get_configuration_space(
#         self, seed: Union[int, None] = None
#     ) -> ConfigSpace.ConfigurationSpace:
#         cs = CS.ConfigurationSpace(seed=seed or np.random.randint(1, 100000))

#         hyperparameters = []
#         self.min_value = {}
#         for key, value in self.config_knob.items():
#             if value["type"] == "enum":
#                 hyperparameters.append(
#                     CS.CategoricalHyperparameter(
#                         key, choices=value["range"], default_value=str(value["default"])
#                     )
#                 )
#             else:
#                 raise TypeError

#         cs.add_hyperparameters(hyperparameters)

#         return cs

#     def get_fidelity_space(
#         self, seed: Union[int, None] = None
#     ) -> ConfigSpace.ConfigurationSpace:
#         return CS.ConfigurationSpace(seed=seed or np.random.randint(1, 100000))

#     def get_meta_information(self) -> Dict:
#         return {"number_objective": len(self.objectives)}


if __name__ == "__main__":
    a = GCCTuning("1", 121, 0, 1)
