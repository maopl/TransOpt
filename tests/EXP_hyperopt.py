from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from transopt.benchmark.HPO.HPO_ERM import HPO_ERM, HPO_ERM_JSD
import numpy as np

# Create a single HPO_ERM instance
hpo = HPO_ERM(task_name='hyperopt_optimization', budget_type='FEs', budget=2000, seed=42, 
                workload=0, algorithm='ERM', gpu_id=3, augment=None, architecture='alexnet', 
                model_size=1, optimizer='hyperopt_without_augment', base_dir='/data2/mpl')  
# Define the objective function
def objective(params):
    # Convert hyperopt params to the format expected by HPO_ERM
    config = np.array([params[name] for name in hpo.configuration_space.variables_order])
    result = hpo.objective_function(configuration=config)
    return {'loss': 1 - result['function_value'], 'status': STATUS_OK}

# Define the search space
def get_hyperopt_space():
    original_ranges = hpo.configuration_space.original_ranges
    space = {}
    for param_name, param_range in original_ranges.items():
        space[param_name] = hp.uniform(param_name, param_range[0], param_range[1])
    return space

if __name__ == "__main__":
    # Create the search space
    search_space = get_hyperopt_space()
    
    # Run optimization
    n_iterations = 200
    trials = Trials()
    # Set a random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    
    best = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=n_iterations,
                trials=trials,
                rstate=np.random.default_rng(random_seed))
    
    # Print results
    print("Best hyperparameters found:", best)
    print("Best objective value:", 1 - min(trials.losses()))
