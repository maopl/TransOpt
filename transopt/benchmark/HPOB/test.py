import numpy as np
import os
import json
import zipfile


# data URL https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip

class TabularOptimizationProblem:
    def __init__(self, zip_path, mode="v3"):
        """
        Constructor to initialize the tabular optimization problem.
        Inputs:
            * zip_path: Path to the zip file containing the benchmark data.
            * mode: Mode name indicating how to load the data. Options: v1, v2, v3.
        """
        self.zip_path = zip_path
        self.mode = mode
        self.load_data()

    def load_data(self):
        """
        Loads the dataset from the provided zip file and mode.
        """
        print("Loading dataset...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open("hpob-data/meta-train-dataset.json") as f:
                self.meta_train_data = json.load(f)
            with zip_ref.open("hpob-data/meta-test-dataset.json") as f:
                self.meta_test_data = json.load(f)
            with zip_ref.open("hpob-data/meta-validation-dataset.json") as f:
                self.meta_validation_data = json.load(f)

        dimensions = {}
        if self.mode in ["v1", "v2", "v3"]:
            temp_data = {}
            for search_space in self.meta_train_data.keys():
                temp_data[search_space] = {}

                for dataset in self.meta_train_data[search_space].keys():
                    temp_data[search_space][dataset] = self.meta_train_data[search_space][dataset]
                    dimensions[f"{search_space}-{dataset}"] = len(self.meta_train_data[search_space][dataset]['X'][0])

                if search_space in self.meta_test_data.keys():
                    for dataset in self.meta_test_data[search_space].keys():
                        temp_data[search_space][dataset] = self.meta_test_data[search_space][dataset]
                        dimensions[f"{search_space}-{dataset}"] = len(self.meta_test_data[search_space][dataset]['X'][0])

                if search_space in self.meta_validation_data.keys():
                    for dataset in self.meta_validation_data[search_space].keys():
                        temp_data[search_space][dataset] = self.meta_validation_data[search_space][dataset]
                        dimensions[f"{search_space}-{dataset}"] = len(self.meta_validation_data[search_space][dataset]['X'][0])

            self.data = temp_data
        else:
            raise ValueError("Provide a valid mode: v1, v2, or v3")

        # Save dimensions to a JSON file
        dimensions_path = "dataset_dimensions.json"
        with open(dimensions_path, "w") as f:
            json.dump(dimensions, f, indent=4)

        print("Dataset loaded and dimensions saved to dataset_dimensions.json.")

        
    def evaluate(self, search_space_id, dataset_id, config_idx):
        """
        Evaluates the given configuration on the selected dataset.
        Inputs:
            * search_space_id: Identifier of the search space for the evaluation.
            * dataset_id: Identifier of the dataset for the evaluation.
            * config_idx: Index of the configuration to evaluate.
        Output:
            * performance: The performance metric of the given configuration.
        """
        assert search_space_id in self.data, "Invalid search space ID"
        assert dataset_id in self.data[search_space_id], "Invalid dataset ID"
        assert 0 <= config_idx < len(self.data[search_space_id][dataset_id]["X"]), "Invalid configuration index"

        performance = self.data[search_space_id][dataset_id]["y"][config_idx]
        return performance

    def get_configuration(self, search_space_id, dataset_id, config_idx):
        """
        Retrieves the configuration for the selected dataset.
        Inputs:
            * search_space_id: Identifier of the search space.
            * dataset_id: Identifier of the dataset.
            * config_idx: Index of the configuration to retrieve.
        Output:
            * configuration: The hyperparameter configuration as a list.
        """
        assert search_space_id in self.data, "Invalid search space ID"
        assert dataset_id in self.data[search_space_id], "Invalid dataset ID"
        assert 0 <= config_idx < len(self.data[search_space_id][dataset_id]["X"]), "Invalid configuration index"

        configuration = self.data[search_space_id][dataset_id]["X"][config_idx]
        return configuration

# Example usage
if __name__ == "__main__":
    # Specify the path to the dataset zip file
    zip_path = "hpob-data.zip"
    # Create an instance of TabularOptimizationProblem
    tabular_problem = TabularOptimizationProblem(zip_path, mode="v3")
    # Evaluate a configuration
    search_space_id = "search_space_id_1"
    dataset_id = "dataset_id_1"
    config_idx = 0
    performance = tabular_problem.evaluate(search_space_id, dataset_id, config_idx)
    print("Performance of configuration:", performance)
    # Retrieve a configuration
    configuration = tabular_problem.get_configuration(search_space_id, dataset_id, config_idx)
    print("Configuration:", configuration)
