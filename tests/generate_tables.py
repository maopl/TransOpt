import random
import string

from transopt.datamanager.database import Database


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
        {"name": f"obj_{i}_{generate_random_string(3)}", "type": "float"}
        for i in range(num_objectives)
    ]
    fidelities = []  # No fidelities defined in your setup, can be adjusted if needed

    return table_name, {
        "variables": variables,
        "objectives": objectives,
        "fidelities": fidelities,
    }


def create_test_tables(db, num_tables):
    for _ in range(num_tables):
        table_name, problem_cfg = generate_table_config()
        db.create_table(table_name, problem_cfg)

if __name__ == "__main__":
    db = Database()  # Assuming Database is properly initialized and can be used

    # 创建测试 datasets
    # create_test_tables(db, 200)

    # 获取所有的 datasets
    print(db.get_table_list())
    
    # 获取某一 dataset 的 info
    # print(db.query_dataset_info('transportation_2_5_fxrvy'))

    db.close()
