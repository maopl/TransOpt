import numpy as np
from transopt.benchmark.HPO.HPO_ERM import HPO_ERM

# Create HPO_ERM instance
hpo = HPO_ERM(task_name='design_optimization', budget_type='FEs', budget=2000, seed=42,
              workload=0, algorithm='ERM', gpu_id=0, augment='cutout', architecture='alexnet',
              model_size=1, optimizer='design', base_dir='/data/')

def evaluate_configurations(configurations):
    results = []
    for i, config in enumerate(configurations):
        # Ensure config matches the order of variables in configuration space
        config_list = np.array([config[name] for name in hpo.configuration_space.variables_order])
        
        result = hpo.objective_function(configuration=config_list)
        objective_value = 1 - result['function_value']
        
        results.append({
            'iteration': i + 1,
            'configuration': config,
            'objective_value': objective_value,
            'results_all': result
        })
        print(f'Configuration {i+1}: objective value = {objective_value:.4f}')
    
    return results

if __name__ == "__main__":
    # Example configurations list (you should replace this with your actual configurations)
    example_configurations = [
        {'lr': -2.319,
        'weight_decay': -4.471,
        # 'lr': 10**(-2.319),
        # 'weight_decay': 10**(-4.471), 
        'momentum': 0.9606669283265548,
        'batch_size': 2,
        'dropout_rate': 0.15964430498134813}
    ]
    
    # Evaluate all configurations
    # results = evaluate_configurations(example_configurations)
    
    # Calculate mean corruption accuracy
    corruption_accs = [
        0.5158,  # gaussian_noise
        0.5258,  # shot_noise 
        0.3818,  # impulse_noise
        0.6208,  # defocus_blur
        0.519,   # glass_blur
        0.611,   # motion_blur
        0.6048,  # zoom_blur
        0.6248,  # snow
        0.5784,  # frost
        0.5038,  # fog
        0.7148,  # brightness
        0.2438,  # contrast
        0.6872,  # elastic_transform
        0.6772,  # pixelate
        0.7362   # jpeg_compression
    ]
    
    mean_corruption_acc = np.mean(corruption_accs)
    print(f"\nMean corruption accuracy: {mean_corruption_acc:.4f}")
    
    # # Find best configuration
    # best_result = max(results, key=lambda x: x['objective_value'])
    # print(f"\nBest configuration found:")
    # print(f"Iteration: {best_result['iteration']}")
    # print(f"Configuration: {best_result['configuration']}")
    # print(f"Objective value: {best_result['objective_value']:.4f}")




