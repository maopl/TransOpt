import os
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem


@probelm_registry.register("Absolut")
class Absolut(NonTabularProblem):