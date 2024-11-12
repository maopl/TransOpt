import gym
import logging
import random
import numpy as np
from typing import Union, Dict

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *


logger = logging.getLogger("LunarLanderBenchmark")

# 计算两组数据的 Pearson 相关系数和 p 值


def lunar_lander_simulation(w, print_reward=False, seed=1, dimension=12):
    total_reward = 0.0
    steps = 0
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    s = env.reset(seed=seed)[0]
    while True:
        a = heuristic_controller(s, w)
        s, r, done, info, _ = env.step(a)
        total_reward += r
        steps += 1
        if done:
            break
    if print_reward:
        print(f"Total reward: {total_reward}")
    return total_reward


def heuristic_controller(s, w, is_continuous=False):
    # w is the array of controller parameters of shape (1, 12)
    angle_target = s[0] * w[0] + s[2] * w[1]
    if angle_target > w[2]:
        angle_target = w[2]
    if angle_target < w[-2]:
        angle_target = -w[2]
    hover_target = w[3] * np.abs(s[0])
    angle_todo = (angle_target - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_target - s[1]) * w[6] - (s[3]) * w[7]
    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]
    if is_continuous:
        a = np.array([hover_todo * 20 - 1, angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
    return a



@problem_registry.register("LunarlanderBenchmark")
class LunarlanderBenchmark(NonTabularProblem):
    problem_type = "LunarlanderBenchmark"
    fidelity = None
    workloads = [0, 1, 2, 3, 4, 5, 6, 7]  
    num_variables = 12
    num_objectives = 1
    def __init__(self, task_name, budget_type, budget, seed, workload, **kwargs):
        self.workload = workload
        self.lunar_seeds = [2, 3, 4, 5, 10, 14, 15, 19]
        self.lunar_seed = self.lunar_seeds[workload]

        super().__init__(task_name=task_name, budget=budget, budget_type=budget_type, 
                        workload=workload, seed=seed)

    def get_configuration_space(self) -> SearchSpace:
        variables = [Continuous(f'x{i}', (0, 2.0)) for i in range(self.num_variables)]
        return SearchSpace(variables)

    def get_fidelity_space(self) -> FidelitySpace:
        return FidelitySpace([])

    def cal_reward(self, X):
        return lunar_lander_simulation(X, seed=self.lunar_seed, dimension=self.num_variables)

    def objective_function(self, configuration: dict, fidelity=None, seed=None, **kwargs):
        """Compute the objective values based on configuration."""
        X = np.array([configuration[f'x{i}'] for i in range(self.num_variables)])

        obj_value = self.cal_reward(X)

        return {
            'reward': float(obj_value)}

    def get_objectives(self) -> dict:
        """Define objectives for optimization: maximizing reward."""
        return {
            'reward': 'maximize'  # Objective: maximize lunar lander reward
        }

    def get_problem_type(self) -> str:
        return self.problem_type

    def get_meta_information(self) -> Dict:
        return {}


if __name__ == "__main__":
    seed_list = [2, 3, 4, 5, 10, 14, 15, 19]
    
