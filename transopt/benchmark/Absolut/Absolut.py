import os
from typing import Dict

from numpy.random.mtrand import RandomState as RandomState
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem


@probelm_registry.register("Absolut")
class Absolut(NonTabularProblem):
    def __init__(
        self, task_name, budget, seed, workload, **kwargs
    ):
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
        seed: RandomState | int | None = None,
        **kwargs,
    ) -> Dict:
        return super().objective_function(configuration, fidelity, seed, **kwargs)