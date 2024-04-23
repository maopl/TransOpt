import numpy as np

from sampler.sampler_base import Sampler
from agent.registry import sampler_register

@sampler_register("random")
class RandomSampler(Sampler):
    def sample(self, search_space, n_samples=1, metadata = None):
        samples = np.zeros((n_samples, len(search_space.variables_order)))
        for i, name in enumerate(search_space.variables_order):
            var_range = search_space.ranges[name]
            if search_space.var_discrete[name]:  # 判断是否为离散变量
                samples[:, i] = np.random.randint(
                    var_range[0], var_range[1] + 1, size=n_samples
                )
            else:
                samples[:, i] = np.random.uniform(
                    var_range[0], var_range[1], size=n_samples
                )
        return samples