import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from transopt.benchmark.HPO.HPO_ERM import HPO_ERM

# Create a single HPO_ERM instance
hpo = HPO_ERM(task_name='hebo_optimization', budget_type='FEs', budget=2000, seed=42, 
                workload=0, algorithm='ERM', gpu_id=0, augment=None, architecture='alexnet', 
                model_size=1, optimizer='hebo_without_augment', base_dir='/data2/mpl')

# Define the objective function
def objective(config):
    # Convert config dict to list according to variables_order
    config_list = np.array([config[name] for name in hpo.configuration_space.variables_order])
    
    result = hpo.objective_function(configuration=config_list)
    return 1 - result['function_value']

# Define the design space
def get_design_space():
    original_ranges = hpo.configuration_space.original_ranges
    space = DesignSpace().parse([
        {'name': param_name, 'type': 'num', 'lb': param_range[0], 'ub': param_range[1]}
        for param_name, param_range in original_ranges.items()
    ])
    return space

if __name__ == "__main__":
    # Create the design space
    design_space = get_design_space()
    
    # Initialize HEBO
    opt = HEBO(design_space, scramble_seed=0)
    
    # Run optimization
    n_iterations = 200
    for i in range(n_iterations):
        rec = opt.suggest(n_suggestions=1)
        f_val = objective(rec.to_dict(orient='records')[0])
        y = np.array([[f_val]])
        opt.observe(rec, y)
        print(f'After {i+1} iterations, best obj is {opt.y.min():.4f}')


