import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from transopt.benchmark.HPO.HPO_ERM import HPO_ERM, HPO_ERM_Para

class HPOProblem(Problem):
    def __init__(self, task_name, budget_type, budget, seed, workload):
        self.hpo = HPO_ERM_Para(task_name=task_name, budget_type=budget_type, budget=budget, 
                           seed=seed, workload=workload, algorithm='ERM_ParaAUG', gpu_id=1, 
                           augment='ParaAUG', architecture='mobilenet', model_size=1, optimizer='nsga2_ddpm',)

        original_ranges = self.hpo.configuration_space.original_ranges
        variables_order = self.hpo.configuration_space.variables_order
        n_var = len(variables_order)
        xl = np.array([original_ranges[var][0] for var in variables_order])
        xu = np.array([original_ranges[var][1] for var in variables_order])
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)
        
    def set_upper_config(self, upper_config):
        self.upper_config = upper_config
    
    def get_upper_space(self):
        return self.hpo.get_upper_space()
    
    
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []
        for x in X:
            config = {**self.upper_config, **dict(zip(self.hpo.configuration_space.variables_order, x))}
            val_acc = self.hpo.objective_function(config)
            f1.append(1 - val_acc['test_standard_acc'])  # Minimize 1 - accuracy
            f2.append(1- val_acc['test_robust_acc'])  # Minimize number of epochs
        out["F"] = np.column_stack([f1, f2])


if __name__ == "__main__":
    # Initialize HPO instance to get spaces
    problem = HPOProblem(task_name='test_task', budget_type='FEs', budget=3000, seed=0, workload=0)

    upper_space = problem.get_upper_space()
    
    # Random search in upper level
    n_upper_samples = 10
    best_upper_config = None
    best_lower_f = float('inf') 
    
    for _ in range(n_upper_samples):
        # Random sample from upper space
        upper_config = {param: np.random.uniform(range[1][0], range[1][1]) 
                       for param, range in upper_space.items()}
        problem.set_upper_config(upper_config)
        
        # Solve lower level problem with current upper config

        algorithm = NSGA2(pop_size=40)
        res = minimize(problem, algorithm, ('n_gen', 30), seed=1, verbose=True)
        
        # Get best lower level objective
        min_f = min(np.sum(res.F, axis=1))
        
        # Update best solution if better
        if min_f < best_lower_f:
            best_lower_f = min_f
            best_upper_config = upper_config
    
    print("Best upper level configuration found:")
    print(best_upper_config)
    print("With lower level objective value:", best_lower_f)
