import os
import numpy as np
from typing import Union, Dict
from numpy.random.mtrand import RandomState as RandomState
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem


@problem_registry.register("Absolut")
class Absolut(NonTabularProblem):
    def __init__(
        self, task_name, budget, seed, workload, **kwargs
    ):
        # 目标晶格抗原
        self.antigens = kwargs["antigens"]
        super(Absolut, self).__init__(
            task_name=task_name,
            budget=budget,
            seed=seed,
            workload=workload,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs
    ) -> Dict:
        a
        return super().objective_function(configuration, fidelity, seed, **kwargs)
