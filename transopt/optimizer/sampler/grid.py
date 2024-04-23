import numpy as np
from sampler.sampler_base import Sampler
from agent.registry import sampler_register

@sampler_register("grid")
class GridSampler(Sampler):
    def generate_grid_for_variable(self, var_range, is_discrete, steps):
        if is_discrete:
            # 如果步长大于离散值的数量，返回所有可能的离散值
            if (var_range[1] - var_range[0] + 1) <= steps:
                return np.arange(var_range[0], var_range[1] + 1)
            else:
                # 否则，均匀选择步长间隔的离散值
                return np.linspace(
                    var_range[0], var_range[1], num=steps, endpoint=True
                ).round()
        else:
            return np.linspace(var_range[0], var_range[1], num=steps)

    def sample(self, search_space, steps=5, metadata=None):
        """
        生成采样点。
        :param search_space: 搜索空间对象。
        :param steps: 控制采样密度的步长，对于离散变量，如果变量的可能值少于步长，则返回所有值。
        :return: 所有采样点的集合。
        """
        grids = []
        for name in search_space.variables_order:
            var_range = search_space.ranges[name]
            is_discrete = search_space.var_discrete[name]
            grid = self.generate_grid_for_variable(var_range, is_discrete, steps)
            grids.append(grid)

        mesh = np.meshgrid(*grids, indexing="ij")
        sample_points = np.stack(mesh, axis=-1).reshape(
            -1, len(search_space.variables_order)
        )
        return sample_points
