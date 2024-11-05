import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from transopt.benchmark.HPO.HPO_ERM import HPO_ERM, HPO_ERM_JSD

class HPOProblem(Problem):
    def __init__(self, task_name, budget_type, budget, seed, workload):
        self.hpo = HPO_ERM(task_name=task_name, budget_type=budget_type, budget=budget, seed=seed, workload=workload, algorithm='ERM', gpu_id=1, augment='ddpm', architecture='wideresnet', model_size=28, optimizer='nsga2_ddpm')

        original_ranges = self.hpo.configuration_space.original_ranges
        variables_order = self.hpo.configuration_space.variables_order
        n_var = len(variables_order)
        xl = np.array([original_ranges[var][0] for var in variables_order])
        xu = np.array([original_ranges[var][1] for var in variables_order])
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)
    
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []
        for x in X:
            val_acc = self.hpo.objective_function(x)
            f1.append(1 - val_acc['test_standard_acc'])  # Minimize 1 - accuracy
            f2.append(1- val_acc['test_robust_acc'])  # Minimize number of epochs
        out["F"] = np.column_stack([f1, f2])

if __name__ == "__main__":
    problem = HPOProblem(task_name='test_task', budget_type='FEs', budget=3000, seed=0, workload=0)
    algorithm = NSGA2(pop_size=40)
    res = minimize(problem, algorithm, ('n_gen', 30), seed=1, verbose=True)
    
    print("Best solutions found:")
    for i in range(len(res.X)):
        print(f"Solution {i+1}: {res.X[i]}, Objectives: {res.F[i]}")
