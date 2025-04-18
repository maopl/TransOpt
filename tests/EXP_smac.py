from ConfigSpace import ConfigurationSpace
import ConfigSpace as cs
import numpy as np
import time
from smac import HyperparameterOptimizationFacade, Scenario
from transopt.benchmark.HPO.HPO_ERM import HPO_ERM

# Create a single HPO_ERM instance
hpo = HPO_ERM(task_name='smac_optimization', budget_type='FEs', budget=2000, seed=42, 
                workload=0, algorithm='ERM', gpu_id=3, augment=None, architecture='alexnet', 
                model_size=1, optimizer='smac_without_augment', base_dir='/data2/mpl')  
# Define the objective function
def objective(configuration, seed: int = 0):
    start = time.time()
    config_list = np.array([configuration.get(name) for name in hpo.configuration_space.variables_order])
    result = hpo.objective_function(configuration=config_list)
    end = time.time()
    return 1 - result['function_value']  # SMAC minimizes, so we return 1 - accuracy

# Define the configuration space
def get_configspace():
    space = ConfigurationSpace()
    original_ranges = hpo.configuration_space.original_ranges
    for param_name, param_range in original_ranges.items():
        space.add_hyperparameter(cs.UniformFloatHyperparameter(param_name, lower=param_range[0], upper=param_range[1]))
    return space

if __name__ == "__main__":
    # Create the configuration space
    config_space = get_configspace()
    
    # Scenario object specifying the optimization environment
    scenario = Scenario(config_space, deterministic=True, n_trials=200)
    
    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, objective)
    incumbent = smac.optimize()
    
    # Print the best configuration and its performance
    print(f"Best configuration: {incumbent}")
    print(f"Best performance: {1 - smac.intensifier.trajectory[-1].cost}")  # Convert back to accuracy
