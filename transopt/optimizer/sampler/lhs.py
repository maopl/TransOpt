import numpy as np
from scipy.stats import qmc

from sampler.sampler_base import Sampler
from agent.registry import sampler_register

@sampler_register.register("lhs")
class LatinHypercubeSampler(Sampler):
    def sample(self, search_space, n_samples=10, metadata = None):
        d = len(search_space.variables_order)
        sampler = qmc.LatinHypercube(d=d)
        sample_points = sampler.random(n=n_samples)
        for i, name in enumerate(search_space.variables_order):
            var_range = search_space.ranges[name]
            if search_space.var_discrete[name]:  # 离散变量处理
                # 离散变量通过四舍五入的方式来采样
                continuous_vals = qmc.scale(
                    sample_points[:, i], var_range[0], var_range[1]
                )
                sample_points[:, i] = np.round(continuous_vals).astype(int)
            else:  # 连续变量处理
                sample_points[:, i] = qmc.scale(
                    sample_points[:, i], var_range[0], var_range[1]
                )
        return sample_points
