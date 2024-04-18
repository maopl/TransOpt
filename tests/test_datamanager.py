import random
import unittest
from unittest.mock import MagicMock

from datamanager.manager import DataManager

combinations = [(2, 2), (3, 1), (1, 3)]
base_vars = [
    ['temperature', 'pressure'],
    ['altitude', 'velocity', 'acceleration'],
    ['humidity'],
    ['brightness', 'color_intensity', 'saturation'],
    ['financial_gain', 'financial_loss', 'financial_net']
]

def generate_variable_name(base_name):
    """Generate a variable name by adding a random suffix or prefix to make it more realistic."""
    suffix = random.choice(['_early', '_late', '_avg'])
    prefix = random.choice(['max_', 'min_', ''])
    return f"{prefix}{base_name}{suffix}"

def create_variables(category):
    """Create a set of variables based on the selected category with extended names."""
    return [{'name': generate_variable_name(var)} for var in base_vars[category]]

def create_datasets(num_ds):
    """Generate a list of documents with structured data."""
    datasets = []
    for _ in range(num_ds):
        num_variables, num_objectives = random.choice(combinations)
        category = random.randint(0, len(base_vars) - 1)
        variables = create_variables(category)
        dataset_info = {
            'num_variables': num_variables,
            'num_objectives': num_objectives,
            'variables': variables
        }
        datasets.append(dataset_info)
    return datasets


class TestDataManager(unittest.TestCase):
    def setUp(self):
        # Mock the database and its methods
        self.mock_db = MagicMock()
        self.mock_db.get_table_list.return_value = [f'dataset{i}' for i in range(1, 1001)]
        self.test_datasets = create_datasets(1000)
        self.mock_db.query_config.side_effect = self.test_datasets

        # Initialize DataManager with the mocked database
        self.data_manager = DataManager(db=self.mock_db, num_hashes=100, char_ngram=5, num_bands=50, random_state=12345)

    def test_lsh_cache_initialization(self):
        # Check if the LSH cache is initialized with the correct number of datasets
        self.assertEqual(len(self.data_manager.lsh_cache.buckets), 10)
        # Validate that datasets are correctly fingerprinted
        self.assertTrue(len(self.data_manager.lsh_cache.fingerprints) >= 1000)

    def test_adding_and_querying_datasets(self):
        # Add a new dataset
        new_dataset_info = create_datasets(1)[0]  # Mimic existing dataset structure
        dataset_name = 'new_dataset'
        self.data_manager.create_dataset(dataset_name, new_dataset_info)
        
        # Query for a similar dataset
        similar_datasets = self.data_manager.get_similar_datasets(dataset_name, new_dataset_info)
        # Check if any dataset is found similar, expected as data structure mimics the existing one
        self.assertTrue(len(similar_datasets) > 0)

    def test_database_integration(self):
        # Check if the new dataset is inserted in the database
        dataset_name = 'test_insertion'
        data_to_insert = [{'x': 1, 'y': 2}]
        self.data_manager.insert_data(dataset_name, data_to_insert)
        self.mock_db.insert_data.assert_called_with(dataset_name, data_to_insert)

if __name__ == '__main__':
    unittest.main()
