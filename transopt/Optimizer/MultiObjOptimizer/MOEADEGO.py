import numpy as np
from sklearn.cluster import KMeans
from pymoo.algorithms.moo.moead import MOEAD as MOEADAlgo
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.optimize import minimize

from transopt.Optimizer.OptimizerBase import BOBase
from transopt.utils.Data import ndarray_to_vectors
from transopt.utils.Register import optimizer_register
from transopt.utils.Normalization import get_normalizer
from transopt.utils.sampling import lhs
from transopt.utils.pareto import find_pareto_front


@optimizer_register("MOEAD")
class MOEADEGO(BOBase):
    def __init__(self, config: dict, **kwargs):
        pass

    def _solve(self, X, Y, batch_size):
        
        # initialize population
        if len(X) < self.pop_size:
            X = np.vstack([X, lhs(X.shape[1], self.pop_size - len(X))])
        elif len(X) > self.pop_size:
            sorted_indices = NonDominatedSorting().do(Y)
            X = X[sorted_indices[:self.pop_size]]
        self.algo.initialization.sampling = X