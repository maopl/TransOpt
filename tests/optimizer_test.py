import json
import random
import unittest
from unittest.mock import MagicMock



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


if __name__ == '__main__':
    unittest.main()