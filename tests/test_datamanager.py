import json
import random
import unittest
from unittest.mock import MagicMock

from transopt.datamanager.manager import DataManager


file_path = __file__.replace('test_datamanager.py', 'test_data.json')

with open(file_path, 'r') as f:
    test_data = json.load(f)

test_data_name = [dataset['dataset_name'] for dataset in test_data]

class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.test_datasets = test_data[:4000]
        self.dataset_names = test_data_name[:4000]
        
        # Mock the database and its methods
        self.mock_db = MagicMock()

        self.mock_db.get_table_list.return_value = self.dataset_names
        
        def query_config_side_effect(dataset_name):
            for dataset in self.test_datasets:
                if dataset['dataset_name'] == dataset_name:
                    return dataset
            return None
        
        self.mock_db.query_config.side_effect = query_config_side_effect

        # Initialize DataManager with the mocked database
        self.data_manager = DataManager(db=self.mock_db, num_hashes=100, char_ngram=8, num_bands=50, random_state=12345)

    def test_lsh_cache_initialization(self):
        # Check if the LSH cache is initialized with the correct number of datasets
        self.assertEqual(len(self.data_manager.lsh_cache.buckets), 10)
        # Validate that datasets are correctly fingerprinted
        self.assertTrue(len(self.data_manager.lsh_cache.fingerprints) >= 1000)

    def test_adding_and_querying_datasets(self):
        # Add a new dataset
        new_dataset_info = test_data[-1]
        dataset_name = test_data_name[-1]
        # self.data_manager.create_dataset(dataset_name, new_dataset_info)
        
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
