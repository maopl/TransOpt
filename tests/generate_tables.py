import random
import string

from transopt.datamanager.database import Database
from transopt.agent.registry import *


base_strings = {
    "finance": [
        "interest_rate",
        "loan_amount",
        "credit_score",
        "investment_return",
        "market_risk",
    ],
    "health": [
        "blood_pressure",
        "heart_rate",
        "cholesterol_level",
        "blood_sugar",
        "body_mass_index",
    ],
    "transportation": [
        "traffic_flow",
        "fuel_usage",
        "travel_time",
        "vehicle_capacity",
        "route_efficiency",
    ],
    "energy": [
        "power_consumption",
        "emission_level",
        "renewable_source",
        "energy_cost",
        "grid_stability",
    ],
    "education": [
        "student_performance",
        "teacher_ratio",
        "course_availability",
        "graduation_rate",
        "facility_utilization",
    ],
}


def generate_random_string(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def generate_comma_separated_numbers(count):
    return ",".join(str(random.randint(1, 100)) for _ in range(count))


def choose_random_from_registry(registry):
    return random.choice(registry.list_names() + ["default"])

def initialize_modules():
    import transopt.benchmark.synthetic
    import transopt.optimizer.acquisition_function
    import transopt.optimizer.model
    import transopt.optimizer.pretrain
    import transopt.optimizer.refiner
    import transopt.optimizer.sampler

def generate_table_config():
    domain = random.choice(list(base_strings.keys()))
    num_variables = random.randint(3, 5)
    num_objectives = random.randint(1, 2)

    dataset_name_suffix = generate_random_string(5)
    table_name = f"{domain}_{num_objectives}_{num_variables}_{dataset_name_suffix}"

    variables = []
    selected_base_strings = random.sample(base_strings[domain], k=num_variables)
    for base in selected_base_strings:
        random_suffix = generate_random_string(random.randint(1, 3))
        variable_name = f"{base}_{random_suffix}"
        variables.append(
            {"name": variable_name, "type": "continuous"}
        )  # Assume all variables are float for simplicity

    objectives = [
        {"name": f"obj_{i}_{generate_random_string(3)}", "type": "minimize"}
        for i in range(num_objectives)
    ]
    fidelities = []  # No fidelities defined in your setup, can be adjusted if needed

    used_dataset = random.sample(list(base_strings.keys()), k=random.randint(1, 3))
    
    initialize_modules()

    # Additional fields
    additional_config = {
        "name": table_name,
        "dim": num_variables,
        "obj": num_objectives,
        "fidelity": generate_random_string(random.randint(3, 6)),
        "workloads": generate_comma_separated_numbers(random.randint(1, 5)),
        "budget_type": random.choice(["Num_FEs", "Hours", "Minutes", "Seconds"]),
        "budget": random.randint(1, 100),
        "seeds": generate_comma_separated_numbers(random.randint(1, 5)),
        "SpaceRefiner": choose_random_from_registry(space_refiner_registry),
        "Sampler": choose_random_from_registry(sampler_registry),
        "Pretrain": choose_random_from_registry(pretrain_registry),
        "Model": choose_random_from_registry(model_registry),
        "ACF": choose_random_from_registry(acf_registry),
        "DatasetSelector": choose_random_from_registry(selector_registry),
        "datasets": used_dataset
    }

    return table_name, {
        "variables": variables,
        "objectives": objectives,
        "fidelities": fidelities,
        **additional_config
    }


def create_test_tables(db, num_tables):
    for _ in range(num_tables):
        table_name, dataset_cfg = generate_table_config()
        db.create_table(table_name, dataset_cfg)

def generate_random_value(data_type):
    if data_type == "continuous":
        return round(random.uniform(0, 100), 2)
    elif data_type == "integer":
        return random.randint(1, 100)

def generate_and_insert_data(db, table_name, dataset_cfg, num_rows=100):
    variables = dataset_cfg["variables"]
    objectives = dataset_cfg["objectives"]
    fidelities = dataset_cfg["fidelities"]
    
    # Generate data
    data = []
    for _ in range(num_rows):
        row = {}
        for var in variables:
            row[var["name"]] = generate_random_value(var["type"])
        for obj in objectives:
            row[obj["name"]] = generate_random_value("continuous")  # Assuming objectives are continuous
        for fid in fidelities:
            row[fid["name"]] = generate_random_value(fid["type"])
        data.append(row)
    
    # Insert data into the database
    db.insert_data(table_name, data)


if __name__ == "__main__":
    db = Database()  # Assuming Database is properly initialized and can be used

    # 创建测试 datasets
    # create_test_tables(db, 200)

    # 获取所有的 datasets
    table_ls = db.get_table_list() 
    # print(table_ls)

    for table in table_ls:
        table_info = db.query_dataset_info(table)
        generate_and_insert_data(db, table, table_info, 100)
    
    # # 获取某一 dataset 的 info
    # table_info = db.query_dataset_info(table_ls[0])

    # generate_and_insert_data(db, table_ls[0], table_info, 100)
    
    # print(db.select_data(table_ls[0]))
    # [{'loan_amount_fyb': 49.57, 'credit_score_el': 60.26, 'market_risk_w': 65.6, 'obj_0_aot': 5.97, 'batch': -1, 'error': 0}, ...]
    
    db.close()
