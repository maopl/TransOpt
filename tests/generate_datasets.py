import random
import string
import time
import uuid
import pandas as pd
from multiprocessing import Manager, Process

from transopt.datamanager.manager import DataManager
from transopt.utils.path import get_library_path

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


def generate_dataset_config():
    domain = random.choice(list(base_strings.keys()))
    num_variables = random.randint(3, 5)
    num_objectives = random.randint(1, 2)

    workload = random.randint(1, 5)
    problem_name = f"{domain}{generate_random_string(3)}"
    dataset_name = f"{problem_name}_{workload}_{uuid.uuid4().hex[:8]}"

    variables = []
    selected_base_strings = random.sample(base_strings[domain], k=num_variables)
    for base in selected_base_strings:
        random_suffix = generate_random_string(random.randint(1, 3))
        variable_name = f"{base}{random_suffix}"
        variables.append(
            {"name": variable_name, "type": "continuous"}
        )  # Assume all variables are float for simplicity

    objectives = [
        {"name": f"obj_{i}_{generate_random_string(3)}", "type": "minimize"}
        for i in range(num_objectives)
    ]
    fidelities = []  # No fidelities defined in your setup, can be adjusted if needed

    # Additional fields
    additional_config = {
        "problem_name": problem_name,
        "dim": num_variables,
        "obj": num_objectives,
        "fidelity": generate_random_string(random.randint(3, 6)),
        "workloads": workload,
        "budget_type": random.choice(["Num_FEs", "Hours", "Minutes", "Seconds"]),
        "budget": random.randint(1, 100),
    }

    return dataset_name, {
        "variables": variables,
        "objectives": objectives,
        "fidelities": fidelities,
        "additional_config": additional_config,
    }


def create_experiment_datasets(dm, num_datasets):
    for _ in range(num_datasets):
        dataset_name, dataset_cfg = generate_dataset_config()
        dm.db.create_table(dataset_name, dataset_cfg, is_experiment=True)
        # dm.create_dataset(dataset_name, dataset_cfg)


def create_exported_datasets(db, num_datasets):
    for _ in range(num_datasets):
        dataset_name, dataset_cfg = generate_dataset_config()
        db.create_table(dataset_name, dataset_cfg, is_experiment=False)


def generate_random_value(data_type):
    if data_type == "continuous":
        return round(random.uniform(0, 100), 2)
    elif data_type == "integer":
        return random.randint(1, 100)


def generate_and_insert_data(db, dataset_name, dataset_cfg, num_rows=100):
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
            row[obj["name"]] = generate_random_value(
                "continuous"
            )  # Assuming objectives are continuous
        for fid in fidelities:
            row[fid["name"]] = generate_random_value(fid["type"])
        data.append(row)

    # Insert data into the database
    db.insert_data(dataset_name, data)


def get_shingles(text, ngram=5):
    return set(text[i : i + ngram] for i in range(len(text) - ngram + 1))


def cal_jacard_similarity(cfg1, cfg2):
    task_name1, variable_names1 = cfg1
    task_name2, variable_names2 = cfg2

    shingles1 = get_shingles(task_name1).union(get_shingles(variable_names1))
    shingles2 = get_shingles(task_name2).union(get_shingles(variable_names2))

    return len(shingles1.intersection(shingles2)) / len(shingles1.union(shingles2))


def validity_experiment(n_tables, num_replicates=3):
    # Clean up the database
    db_path = get_library_path() / "database.db"
    if db_path.exists():
        db_path.unlink()

    dm = DataManager(num_hashes=100, char_ngram=5, num_bands=50)
    setup_start = time.time()
    create_experiment_datasets(dm, n_tables)
    setup_end = time.time()
    print(f"Generated {n_tables} datasets in {setup_end - setup_start} seconds")

    exec_time_jacard = []
    exec_time_lsh = []
    for _ in range(num_replicates):
        target_dataset_name, target_dataset_cfg = generate_dataset_config()
        print(
            f"Searching for similar datasets to {target_dataset_name} with config {target_dataset_cfg}"
        )
        print("=====================================")

        task_name, var_names, num_var, num_obj = dm._construct_vector(
            target_dataset_cfg
        )

        jacard_lower_bound = 0.35

        start_jacard = time.time()
        similar_datasets_by_jacard = set()
        all_datasets = dm.get_all_datasets()
        for dataset in all_datasets:
            dataset_info = dm.get_dataset_info(dataset)
            task_name_tmp, var_names_tmp, num_var_tmp, num_obj_tmp = (
                dm._construct_vector(dataset_info)
            )
            if num_var != num_var_tmp or num_obj != num_obj_tmp:
                continue

            similarity = cal_jacard_similarity(
                (task_name, var_names), (task_name_tmp, var_names_tmp)
            )

            if similarity >= jacard_lower_bound:
                similar_datasets_by_jacard.add(dataset)

        end_jacard = time.time()
        exec_time_jacard.append(end_jacard - start_jacard)
        print(
            f"Found {len(similar_datasets_by_jacard)} similar datasets by jacard in {end_jacard - start_jacard} seconds"
        )

        start_lsh = time.time()
        similar_datasets = dm.search_similar_datasets(target_dataset_cfg)
        similar_datasets_by_lsh = set()
        for dataset in similar_datasets:
            dataset_info = dm.get_dataset_info(dataset)
            task_name_tmp, var_names_tmp, num_var_tmp, num_obj_tmp = (
                dm._construct_vector(dataset_info)
            )
            similarity = cal_jacard_similarity(
                (task_name, var_names), (task_name_tmp, var_names_tmp)
            )

            if similarity >= jacard_lower_bound:
                similar_datasets_by_lsh.add(dataset)

        end_lsh = time.time()
        exec_time_lsh.append(end_lsh - start_lsh)
        print(
            f"Found {len(similar_datasets_by_lsh)} similar datasets by lsh in {end_lsh - start_lsh} seconds"
        )

    dm.teardown()
    return sum(exec_time_jacard) / num_replicates, sum(exec_time_lsh) / num_replicates


if __name__ == "__main__":
    results = []
    num_replicates = 3
    n_tables_list = [1000]
    for n_tables in n_tables_list:
        exec_time_jacard, exec_time_lsh = validity_experiment(n_tables, num_replicates)
        results.append(
            {
                "n_tables": n_tables,
                "exec_time_jacard": exec_time_jacard,
                "exec_time_lsh": exec_time_lsh,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("jacard_vs_lsh.csv", index=False)
