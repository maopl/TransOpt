import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from transopt.benchmark.HPO.HPO_ERM import HPO_ERM, HPO_ERM_Para, HPO_ERM_TestAugment

class HPOProblem(Problem):
    def __init__(self, task_name, budget_type, budget, seed, workload):
        self.hpo = HPO_ERM_TestAugment(task_name=task_name, budget_type=budget_type, budget=budget, 
                           seed=seed, workload=workload, algorithm='ERM', gpu_id=0, 
                           augment='testaugment', architecture='resnet', model_size=18, optimizer='bilevel_random', base_dir='/data/')
        
        # Define the objective function
        original_ranges = self.hpo.configuration_space.original_ranges
        variables_order = self.hpo.configuration_space.variables_order
        n_var = len(variables_order)
        xl = np.array([original_ranges[var][0] for var in variables_order])
        xu = np.array([original_ranges[var][1] for var in variables_order])
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)
        
    def set_policy(self, policy_idx):
        self.hpo.set_policy = policy_idx
    
    def get_policy_space(self):
        return self.hpo.get_subpolicy_space()
    
    def get_random_policy(self):
        return self.hpo.get_random_policy()
    
    
    def _evaluate(self, X, *args, **kwargs):
        f1 = []
        f2 = []
        for x in X:
            config = {**dict(zip(self.hpo.configuration_space.variables_order, x))}
            val_acc = self.hpo.objective_function(config)
            f1.append(1 - val_acc['test_standard_acc'])  # Minimize 1 - accuracy
            f2.append(1- val_acc['test_robust_acc'])  # Minimize number of epochs


if __name__ == "__main__":
    # Initialize HPO instance to get spaces
    problem = HPOProblem(task_name='test_task', budget_type='FEs', budget=3000, seed=0, workload=0)
    
    # # First sampling
    # X1 = np.array([[-2, -4, 0.75, 2, 0.1]])  # [lr, weight_decay, momentum, batch_size]
    # for i in range(problem.get_policy_space()):
    #     # Random sample from upper space
    #     problem.set_policy(i)
    #     problem._evaluate(X1)
    
    # Sample 400 times with random policies and X values
    for i in range(400):
        # Randomly select a policy
        
        # Random sample from upper space
        X2 = np.random.uniform(problem.xl, problem.xu, size=(1, problem.n_var))
        problem.get_random_policy()
        print(problem.hpo.sampler.policy_idx)
        print(problem.hpo.sampler)
        problem._evaluate(X2)
       
