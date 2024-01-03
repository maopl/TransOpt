import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)

data_path = Path(__file__).parent.absolute() / "collected_results" 

def load_and_prepare_data(file_path, objectives):
    """
    Loads JSON data and prepares a DataFrame.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    input_vectors = data["input_vector"]
    output_vectors = data["output_value"]

    df_input = pd.DataFrame(input_vectors)

    df_output = pd.DataFrame(output_vectors)[objectives]
    df_combined = pd.concat([df_input, df_output], axis=1)
    print(f"Loaded {len(df_combined)} data points")

    df_combined = df_combined.drop_duplicates(subset=df_input.columns.tolist())
    print(f"Removed {len(df_combined) - len(df_input)} duplicates")

    for obj in objectives:
        df_combined = df_combined[df_combined[obj] != 1e10]
    print(f"Loaded {len(df_combined)} data points after removing extreme values")
    return df_combined


def calculate_feature_importances(df, objective):
    """
    Calculates and returns feature importances.
    """
    X = df.drop([objective], axis=1)
    y = df[objective]

    model = DecisionTreeRegressor()
    model.fit(X, y)
    feature_importances = model.feature_importances_

    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances}
    )
    return feature_importance_df


def aggregate_importances(importances_list):
    """
    Aggregates a list of importance dataframes by taking the mean of importance scores across all repetitions.
    """
    combined_importances = pd.concat(importances_list)
    mean_importances = combined_importances.groupby("Feature").mean().reset_index()
    return mean_importances.sort_values(by="Importance", ascending=False)


def combine_and_rank_features(importances_list):
    """
    Combines feature importance dataframes and ranks features by total importance across all objectives.
    """
    combined = pd.concat(importances_list)
    combined = (
        combined.groupby("Feature")
        .sum()
        .sort_values(by="Importance", ascending=False)
        .reset_index()
    )
    return combined


def get_top_combined_features(common_features, combined_ranked, total_features=20):
    """
    Supplements the common features with additional features from the combined ranking to reach the desired total.
    """
    final_features = list(common_features)

    # Add more features from the combined ranked list until you reach 20
    for feature in combined_ranked["Feature"]:
        if len(final_features) < total_features:
            if (
                feature not in common_features
            ):  # Only add if not already in common_features
                final_features.append(feature)
        else:
            break  # Stop if we have already 20 features

    return final_features


def find_common_features(importances_list):
    """
    Finds the intersection of important features from multiple importance dataframes.
    """
    top_feature_sets = []

    for df in importances_list:
        # Sort by importance and select the top 20 features
        top_features = df.sort_values(by="Importance", ascending=False).head(20)
        # Add the set of top 20 feature names to the list
        top_feature_sets.append(set(top_features["Feature"]))

    # Find intersection of all top feature sets
    common_features = set.intersection(*top_feature_sets)
    return list(common_features)


def train_and_evaluate_model(
    df, features, objective, use_top_features=False, random_state=42
):
    """
    Trains and evaluates a model, either using top 20 features or all features.
    """
    X = df[features["Feature"]] if use_top_features else df.drop([objective], axis=1)
    y = df[objective]

    # Split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    nrmse = np.sqrt(mean_squared_error(y_test, y_pred)) / np.std(y_test)
    feature_set = "Top 20 Features" if use_top_features else "All Features"
    print(f"{feature_set} - Normalized RMSE: {nrmse}")

    # Get and sort feature importances
    feature_importances = model.feature_importances_
    sorted_features = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    # print("Sorted Feature Importances:")
    # print(sorted_features)

    return nrmse


def get_workloads_improved():
    """
    Returns a list of workloads that improved when including objectives.
    """
    iterations = 5
    workloads_improved = []
    for file in data_path.glob("*.json"):
        workload = file.name.split(".")[0][4:]
        print("==================================================")
        print(workload)
        print("==================================================")

        # Initialize lists to store the results of repeated experiments
        nrmse_excluding_list = []
        nrmse_including_list = []

        for i in range(iterations):
            random_state = 42 + i
            print(f"Running iteration {i+1}/{iterations}...")

            # Repeat the experiment for 'excluding objectives'
            print("CART with top 20 features, excluding objectives")
            df_combined = load_and_prepare_data(file, objectives=["execution_time"])
            important_features = calculate_feature_importances(
                df_combined, "execution_time"
            )
            nrmse_excluding = train_and_evaluate_model(
                df_combined,
                important_features,
                "execution_time",
                use_top_features=True,
                random_state=random_state,
            )
            nrmse_excluding_list.append(nrmse_excluding)
            print("\n")

            # Repeat the experiment for 'including objectives'
            print("CART with top 20 features, including objectives")
            df_combined = load_and_prepare_data(
                file, objectives=["execution_time", "file_size", "compilation_time"]
            )
            important_features = calculate_feature_importances(
                df_combined, "execution_time"
            )
            nrmse_including = train_and_evaluate_model(
                df_combined,
                important_features,
                "execution_time",
                use_top_features=True,
                random_state=random_state,
            )
            nrmse_including_list.append(nrmse_including)
            print("\n")

        # Calculate average or median NRMSE for both configurations
        avg_nrmse_excluding = np.mean(nrmse_excluding_list)
        avg_nrmse_including = np.mean(nrmse_including_list)

        # Compare and record improvements
        if avg_nrmse_including < avg_nrmse_excluding:
            workloads_improved.append(workload)
        print(f"Average Improvement: {avg_nrmse_excluding - avg_nrmse_including}")
        print("\n\n")

    print(f"Workloads improved: {workloads_improved}")

    return workloads_improved


def get_features_for_exp(workloads, repetitions=5):
    features_by_workload = {}

    for workload in workloads:
        print("==================================================")
        print(workload)
        print("==================================================")
        data_file = data_path / f"GCC_{workload}.json"
        features_by_workload[workload] = {}

        # Calculate feature importances for each objective
        importances_et_all, importances_ct_all, importances_fs_all = [], [], []
        for _ in range(repetitions):
            # Repeat the experiment and append the results
            df_combined = load_and_prepare_data(
                data_file, objectives=["execution_time"]
            )
            importances_et_all.append(
                calculate_feature_importances(df_combined, "execution_time")
            )

            df_combined = load_and_prepare_data(
                data_file, objectives=["compilation_time"]
            )
            importances_ct_all.append(
                calculate_feature_importances(df_combined, "compilation_time")
            )

            df_combined = load_and_prepare_data(data_file, objectives=["file_size"])
            importances_fs_all.append(
                calculate_feature_importances(df_combined, "file_size")
            )

        # Aggregate the importances from all repetitions
        importances_et = aggregate_importances(importances_et_all)
        importances_ct = aggregate_importances(importances_ct_all)
        importances_fs = aggregate_importances(importances_fs_all)

        # Find common features across all objectives
        common_features = find_common_features(
            [importances_et, importances_ct, importances_fs]
        )
        print("Top 20 Features (Common):")
        print(common_features)
        features_by_workload[workload]["common"] = common_features

        # Combine and rank features by total importance across all objectives
        combined_ranked = combine_and_rank_features(
            [importances_et, importances_ct, importances_fs]
        )

        # Get top combined features, ensuring we have 20 total
        top_features = get_top_combined_features(common_features, combined_ranked)

        print("Top 20 Features (Common + Supplemented):")
        print(top_features)
        features_by_workload[workload]["top"] = top_features

        # Write feature importances to file

    with open("features_by_workload.json", "w") as fp:
        json.dump(features_by_workload, fp, indent=4)

    print("Features by workload written to features_by_workload.json")


if __name__ == "__main__":
    if not data_path.exists():
        print("Data directory not found. Untarring data...")
        tar = tarfile.open("collected_results.tar.gz")
        tar.extractall()
        tar.close()

    # workloads_improved = get_workloads_improved()

    # workloads_improved = [
    #     "cbench-consumer-tiff2bw",
    #     "cbench-security-rijndael",
    #     "cbench-security-pgp",
    #     "polybench-cholesky",
    #     "cbench-telecom-crc32",
    #     "polybench-fdtd-apml",
    #     "cbench-network-patricia",
    #     "cbench-consumer-tiff2rgba",
    #     "polybench-symm",
    #     "cbench-automotive-susan-e",
    #     "cbench-telecom-adpcm-d",
    #     "polybench-ludcmp",
    #     "polybench-lu",
    #     "cbench-consumer-mad",
    #     "cbench-automotive-qsort1",
    #     "polybench-bicg",
    #     "cbench-security-sha",
    #     "cbench-consumer-jpeg-d",
    #     "cbench-telecom-adpcm-c",
    #     "cbench-telecom-gsm",
    # ]

    # get_features_for_exp(workloads_improved)
    

    # For temp test
    with open(data_path / "GCC_cbench-consumer-tiff2bw.json", "r") as f:
        data = json.load(f)

    input_vectors = data["input_vector"]
    output_vectors = data["output_value"]

    target_objs = ["execution_time", "file_size", "compilation_time"]
    
    # Show target objs in each output vector
    for output_vector in output_vectors:
        for target_obj in target_objs:
            print(f"{target_obj}: {output_vector[target_obj]}")
        print("")
