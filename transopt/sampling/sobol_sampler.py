import numpy as np
from scipy.stats import qmc

from .sampler import Sampler


class SobolSampler(Sampler):
    def sample(self, search_space, n_samples=10):
        d = len(search_space.variables_order)
        sampler = qmc.Sobol(d=d, scramble=True)
        sample_points = sampler.random(n=n_samples)
        for i, name in enumerate(search_space.variables_order):
            var_range = search_space.ranges[name]
            if search_space.var_discrete[name]:
                # 对离散变量进行处理
                continuous_vals = qmc.scale(
                    sample_points[:, i], var_range[0], var_range[1]
                )
                sample_points[:, i] = np.round(continuous_vals).astype(int)
            else:
                sample_points[:, i] = qmc.scale(
                    sample_points[:, i], var_range[0], var_range[1]
                )
        return sample_points
