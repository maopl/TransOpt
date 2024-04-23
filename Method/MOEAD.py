'''
MOEA/D multi-objective solver.
'''

import numpy as np
from sklearn.cluster import KMeans
from pymoo.algorithms.moo.moead import MOEAD as MOEADAlgo
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.optimize import minimize

from optimizer.sampler import lhs_BAK
from transopt.utils.pareto import find_pareto_front
from autooed.mobo.solver.base import Solver


class MOEAD(Solver):
    '''
    Solver based on MOEA/D in MOEA/D-EGO.
    NOTE: only compatible with Direct selection.
    '''
    def __init__(self, problem, n_gen=100, pop_size=100, **kwargs):
        self.real_problem = problem # real problem
        self.problem = None # surrogate problem
        self.transformation = problem.transformation
        self.n_gen = n_gen
        self.pop_size = pop_size

        # generate direction vectors by random sampling
        self.ref_dirs = np.random.random((pop_size, problem.n_obj))
        self.ref_dirs /= np.expand_dims(np.sum(self.ref_dirs, axis=1), 1)
        self.algo = MOEADAlgo(pop_size=pop_size, ref_dirs=self.ref_dirs, eliminate_duplicates=False)

    def solve(self, X, Y, batch_size, acquisition):
        '''
        Solve the multi-objective problem and propose a batch of candidates.

        Parameters
        ----------
        X: np.array
            Current design variables (raw).
        batch_size: int
            Size of the candidate batch.

        Returns
        -------
        X_candidate: np.array
            Proposed candidate design variables (raw).
        Y_candidate: np.array
            Objective values of proposed candidate designs.
        '''
        self.problem = SurrogateProblem(self.real_problem, acquisition)
        X = self.transformation.do(X)
        X_candidate, Y_candidate = self._solve(X, Y, batch_size)
        X_candidate = self.transformation.undo(X_candidate)
        return X_candidate, Y_candidate
    
    def _solve(self, X, Y, batch_size):

        # initialize population
        if len(X) < self.pop_size:
            X = np.vstack([X, lhs_BAK(X.shape[1], self.pop_size - len(X))])
        elif len(X) > self.pop_size:
            sorted_indices = NonDominatedSorting().do(Y)
            X = X[sorted_indices[:self.pop_size]]
        self.algo.initialization.sampling = X

        res = minimize(self.problem, self.algo, ('n_gen', self.n_gen))

        X_candidate, Y_candidate, algo = res.pop.get('X'), res.pop.get('F'), res.algorithm
        G = Y_candidate

        _, curr_pset_idx = find_pareto_front(Y, return_index=True)
        curr_pset = X[curr_pset_idx]

        G_s = algo._decomposition.do(G, weights=self.ref_dirs, ideal_point=algo.ideal_point) # scalarized acquisition value

        # build candidate pool Q
        Q_x, Q_dir, Q_g, Q_gs = [], [], [], []
        X_added = curr_pset.copy()
        for x, ref_dir, g, gs in zip(X_candidate, self.ref_dirs, G, G_s):
            if (x != X_added).any(axis=1).all():
                Q_x.append(x)
                Q_dir.append(ref_dir)
                Q_g.append(g)
                Q_gs.append(gs)
                X_added = np.vstack([X_added, x])
        Q_x, Q_dir, Q_g, Q_gs = np.array(Q_x), np.array(Q_dir), np.array(Q_g), np.array(Q_gs)

        min_batch_size = min(batch_size, len(Q_x)) # in case Q is smaller than batch size

        if min_batch_size == 0:
            indices = np.random.choice(len(X_candidate), batch_size, replace=False)
            return X_candidate[indices], Y_candidate[indices]
        
        # k-means clustering on X with weight vectors
        labels = KMeans(n_clusters=batch_size).fit_predict(np.column_stack([Q_x, Q_dir]))

        # select point in each cluster with lowest scalarized acquisition value
        X_candidate, Y_candidate = [], []
        for i in range(batch_size):
            indices = np.where(labels == i)[0]
            top_idx = indices[np.argmin(Q_gs[indices])]
            X_candidate.append(Q_x[top_idx])
            Y_candidate.append(Q_g[top_idx])

        return np.array(X_candidate), np.array(Y_candidate)