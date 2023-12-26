import time
import logging
import ConfigSpace
import numpy as np
import ConfigSpace as CS
from pathlib import Path
from typing import Union, Dict, Tuple
from csstuning.compiler.compiler_benchmark import GCCBenchmark, LLVMBenchmark

from transopt.Benchmark.BenchBase import NonTabularOptBenchmark
from transopt.utils.Register import benchmark_register

ERROR_VALUE = 1e10


@benchmark_register("GCC")
class GCCTuning(NonTabularOptBenchmark):
    def __init__(
        self, task_name, budget, seed, task_type="non-tabular", workload=None, knobs=None, **kwargs
    ):
        self.objectives = ["execution_time", "file_size", "compilation_time"]
        
        self.workload = workload if workload is not None else GCCBenchmark.AVAILABLE_WORKLOADS[0]
        self.benchmark = GCCBenchmark(workload=self.workload)
        
        all_knobs = self.benchmark.config_space.get_all_details()
        self.config_knob = {k: all_knobs[k] for k in knobs} if knobs else all_knobs
        
        super().__init__(task_name=task_name, seed=seed, task_type=task_type, budget=budget)
        np.random.seed(seed)

    def objective_function(
        self,
        configuration: Union[ConfigSpace.Configuration, Dict],
        fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs
    ) -> Dict:
        c = {}
        for k, v in configuration.items():
            assert k in self.config_knob, f"Invalid configuration key: {k}"
            knob_type = self.config_knob[k]["type"]
            if knob_type == "enum":
                c[k] = self.config_knob[k]["range"][int(v)]
            elif knob_type == "integer":
                c[k] = int(v)

        try:
            start_time = time.time()
            performance = self.benchmark.run(c)
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
                "cost": float(end_time - start_time),
                "info": {"fidelity": fidelity},
            }

            results.update({
                f"function_value_{i + 1}": results[objective]
                for i, objective in enumerate(self.objectives)
            })

            return results
            
        except:
            error_results = {
                "avrg_exec_time": float(ERROR_VALUE),
                "file_size": float(ERROR_VALUE),
                "PAPI_TOT_CYC": float(ERROR_VALUE),
                "PAPI_TOT_INS": float(ERROR_VALUE),
                "PAPI_BR_MSP": float(ERROR_VALUE),
                "PAPI_BR_PRC": float(ERROR_VALUE),
                "PAPI_BR_CN": float(ERROR_VALUE),
                "PAPI_MEM_WCY": float(ERROR_VALUE),
                "maxrss": float(ERROR_VALUE),
                "compilation_time": float(ERROR_VALUE),
                "execution_time": float(ERROR_VALUE),
                "cost": float(end_time - start_time),
                "info": {"fidelity": fidelity},
            }

            error_results.update({
                f"function_value_{i + 1}": error_results[objective]
                for i, objective in enumerate(self.objectives)
            })
            return error_results
        

    def get_configuration_space(
        self, seed: Union[int, None] = None
    ) -> ConfigSpace.ConfigurationSpace:
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

        cs.add_hyperparameters(hyperparameters)

        return cs

    def get_fidelity_space(
        self, seed: Union[int, None] = None
    ) -> ConfigSpace.ConfigurationSpace:
        return CS.ConfigurationSpace(seed=seed or np.random.randint(1, 100000))

    def get_meta_information(self) -> Dict:
        return {"number_objective": len(self.objectives)}


@benchmark_register("LLVM")
class LLVMTuning(NonTabularOptBenchmark):
    def __init__(
        self, task_name, budget, seed, task_type="non-tabular", workload=None, **kwargs
    ):
        if workload is None:
            workload = GCCBenchmark.AVAILABLE_WORKLOADS[0]

        self.benchmark_name = workload
        self.benchmark = LLVMBenchmark(workload=self.benchmark_name)
        self.config_knob = self.benchmark.config_space.get_all_details()
        super(LLVMTuning, self).__init__(
            task_name=task_name,
            seed=seed,
            task_type=task_type,
            budget=budget,
        )
        np.random.seed(seed)

    def objective_function(
        self,
        configuration: Union[ConfigSpace.Configuration, Dict],
        fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs
    ) -> Dict:
        c = {}
        for k, v in configuration.items():
            assert k in self.config_knob
            if self.config_knob[k]["type"] == "enum":
                c[k] = self.config_knob[k]["range"][v]
                configuration[k] = int(configuration[k])
            else:
                raise TypeError

        try:
            start_time = time.time()
            performance = self.benchmark.run(c)
            end_time = time.time()

            return {
                "function_value_1": float(performance["avrg_exec_time"]),
                "function_value_2": float(performance["file_size"]),
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
                "cost": float(end_time - start_time),
                "info": {"fidelity": fidelity},
            }
        except:
            return {
                "function_value_1": float(ERROR_VALUE),
                "function_value_2": float(ERROR_VALUE),
                "avrg_exec_time": float(ERROR_VALUE),
                "file_size": float(ERROR_VALUE),
                "PAPI_TOT_CYC": float(ERROR_VALUE),
                "PAPI_TOT_INS": float(ERROR_VALUE),
                "PAPI_BR_MSP": float(ERROR_VALUE),
                "PAPI_BR_PRC": float(ERROR_VALUE),
                "PAPI_BR_CN": float(ERROR_VALUE),
                "PAPI_MEM_WCY": float(ERROR_VALUE),
                "maxrss": float(ERROR_VALUE),
                "compilation_time": float(ERROR_VALUE),
                "execution_time": float(ERROR_VALUE),
                "cost": float(end_time - start_time),
                "info": {"fidelity": fidelity},
            }

    def get_configuration_space(
        self, seed: Union[int, None] = None
    ) -> ConfigSpace.ConfigurationSpace:
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)

        config = self.benchmark.config_space.get_all_details()
        hyperparameters = []
        expflag = {}
        self.min_value = {}
        for key, value in config.items():
            if value["type"] == "enum":
                hyperparameters.append(
                    CS.CategoricalHyperparameter(
                        key, choices=value["range"], default_value=str(value["default"])
                    )
                )
            else:
                raise TypeError

        cs.add_hyperparameters(hyperparameters)

        return cs

    def get_fidelity_space(
        self, seed: Union[int, None] = None
    ) -> ConfigSpace.ConfigurationSpace:
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        return {"number_objective": 2}


if __name__ == "__main__":
    a = GCCTuning("1", 121, 0, 1)
