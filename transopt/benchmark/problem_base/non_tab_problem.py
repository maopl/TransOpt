""" Base-class of configuration optimization benchmarks """
import os
import json
import logging
import numpy as np
from typing import Union, Dict, List
from pathlib import Path
from transopt.benchmark.problem_base.base import ProblemBase
logger = logging.getLogger("NonTabularProblem")


class NonTabularProblem(ProblemBase):
    def __init__(
        self,
        task_name: str,
        budget_type,
        budget: int,
        workload,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ):
        self.task_name = task_name
        self.budget = budget
        self.workload = workload
        self.lock_flag = False

        super(NonTabularProblem, self).__init__(seed, **kwargs)


    def get_budget(self) -> int:
        """Provides the function evaluations number about the benchmark.

        Returns
        -------
        int
            some human-readable information

        """
        return self.budget

    def get_name(self) -> str:
        """Provides the task name about the benchmark.

        Returns
        -------
        str
            some human-readable information

        """
        return self.task_name

    def get_type(self) -> str:
        """Provides the task type about the benchmark.

        Returns
        -------
        str
            some human-readable information

        """
        return self.task_type

    def get_input_dim(self) -> int:
        """Provides the input dimension about the benchmark.

        Returns
        -------
        int
            some human-readable information

        """
        return self.input_dim

    def get_objective_num(self) -> int:
        return self.num_objective

    def lock(self):
        self.lock_flag = True

    def unlock(self):
        self.lock_flag = False

    def get_lock_state(self) -> bool:
        return self.lock_flag