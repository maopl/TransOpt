import random
import string
import json

# 定义五个不同领域的基础字符串集合
base_strings = {
    'finance': ['interest_rate', 'loan_amount', 'credit_score', 'investment_return', 'market_risk'],
    'health': ['blood_pressure', 'heart_rate', 'cholesterol_level', 'blood_sugar', 'body_mass_index'],
    'transportation': ['traffic_flow', 'fuel_usage', 'travel_time', 'vehicle_capacity', 'route_efficiency'],
    'energy': ['power_consumption', 'emission_level', 'renewable_source', 'energy_cost', 'grid_stability'],
    'education': ['student_performance', 'teacher_ratio', 'course_availability', 'graduation_rate', 'facility_utilization']
}


def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# 生成 5000 条数据
data = []
for _ in range(5000):
    domain = random.choice(list(base_strings.keys()))
    num_variables = random.randint(3, 5)
    num_objectives = random.randint(1, 2)
    
    dataset_name_suffix = generate_random_string(random.randint(1, 3))
    dataset_name = f"{domain}_{num_objectives}_{num_variables}_{dataset_name_suffix}"
    
    variables = []
    selected_base_strings = random.sample(base_strings[domain], k=num_variables)
    for i, base in enumerate(selected_base_strings):
        random_suffix = generate_random_string(random.randint(1, 3))
        variable_name = f"{base}_{random_suffix}"
        variables.append({"name": variable_name})
    
    entry = {
        'dataset_name': dataset_name,
        'num_variables': num_variables,
        'num_objectives': num_objectives,
        'variables': variables
    }
    data.append(entry)
    
file_path = __file__.replace('generate_data.py', 'test_data.json')

with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)
