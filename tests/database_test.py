import os
import sys
import unittest
from pathlib import Path

current_dir = Path(__file__).resolve().parent
package_dir = current_dir.parent
sys.path.insert(0, str(package_dir))

import numpy as np
import pandas as pd
from transopt.KnowledgeBase.database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database("test_database.db")
        self.table_name = "test_table"
        self.dataset_info = {
            "var_names": ["x1", "x2"],
            "var_num": 2,
            "variables": {
                "x1": {"type": "continuous", "range": [-5.12, 5.12], "default": 0.0},
                "x2": {"type": "continuous", "range": [-5.12, 5.12], "default": 0.0},
            },
            
            "obj_names": ["y1", "y2"],
            "obj_num": 2,
            "objectives": {"y1": {"type": "minimize"}, "y2": {"type": "maximize"}},

            "fidelity_names": ["f1", "f2"],
            "fidelity_num": 2,
            "fidelity": {
                "f1": {"type": "continuous", "range": [0, 1], "default": 0.0},
                "f2": {"type": "continuous", "range": [0, 1], "default": 0.0},
            },
        }

        # Table creation
        self.db.create_table(self.table_name, self.dataset_info, overwrite=True)

    def tearDown(self):
        self.db.close()
        os.remove(self.db.data_path)
    
    def assertDictEqual(self, dict1, dict2, msg=None):
        """Helper method to assert that two dictionaries are equal."""
        for key in dict2:
            self.assertEqual(dict1[key], dict2[key], msg)
    
    def assertDictListEqual(self, list1, list2, msg=None):
        """Helper method to assert that two lists of dictionaries are equal."""
        self.assertEqual(len(list1), len(list2), msg)
        for dict1, dict2 in zip(list1, list2):
            self.assertDictEqual(dict1, dict2, msg)

    def test_01_table_creation_and_removal(self):
        """Test the creation and removal of a table."""
        self.assertTrue(self.db.check_table_exist(self.table_name))

        # Test table removal
        self.db.remove_table(self.table_name)
        self.assertFalse(self.db.check_table_exist(self.table_name))

    def test_02_config_storage(self):
        """Test if the configuration is correctly stored."""
        stored_config = self.db.query_config(self.table_name)
        self.assertEqual(stored_config, self.dataset_info)

    def test_03_insert_single_row_dict(self):
        """Test inserting a single row with a dictionary."""
        row = {'x1': 1.0, 'x2': 1.2, 'y1': None, 'y2': None, 'f1': 0.5, 'f2': 0.5}
        rowid = self.db.insert_data(self.table_name, row)[0]
        selected_data = self.db.select_data(self.table_name, rowid=rowid)
        self.assertDictListEqual(selected_data, [row])

    def test_04_insert_multiple_rows_list(self):
        """Test inserting and selecting multiple rows with a list."""
        rows = [
            {'x1': 1.0, 'x2': 1.2, 'y1': None, 'y2': None, 'f1': 0.5, 'f2': 0.5},
            {'x1': 2.0, 'x2': 2.2, 'y1': None, 'y2': None, 'f1': 0.6, 'f2': 0.6}
        ]
        rowids = self.db.insert_data(self.table_name, rows)
        selected_data = self.db.select_data(self.table_name, rowid=rowids)
        self.assertDictListEqual(selected_data, rows)
        
    def test_05_insert_with_dataframe(self):
        """Test inserting and selecting data with a pandas DataFrame."""
        df = pd.DataFrame({
            'x1': [1.0, 2.0],
            'x2': [1.2, 2.2],
            'y1': [None, None],
            'y2': [None, None],
            'f1': [0.5, 0.6],
            'f2': [0.5, 0.6]
        })
        rowids = self.db.insert_data(self.table_name, df)
        selected_data = [self.db.select_data(self.table_name, rowid=rowid)[0] for rowid in rowids]
        expected_data = df.to_dict('records')
        self.assertDictListEqual(selected_data, expected_data)
        
    def test_06_insert_with_ndarray(self):
        """Test inserting and selecting data with a numpy ndarray."""
        arr = np.array([
            [1.0, 1.2, None, None, 0.5, 0.5],
            [2.0, 2.2, None, None, 0.6, 0.6]
        ])
        columns = ['x1', 'x2', 'y1', 'y2', 'f1', 'f2']
        df = pd.DataFrame(arr, columns=columns)
        rowids = self.db.insert_data(self.table_name, df)
        selected_data = self.db.select_data(self.table_name, rowid=rowids)
        expected_data = df.to_dict('records')
        self.assertDictListEqual(selected_data, expected_data)
   
    def test_07_select_data_with_conditions(self):
        # Insert multiple rows of data
        data = [
            {'x1': 1.0, 'x2': 1.2, 'y1': 3.0, 'y2': 4.0, 'f1': 0.5, 'f2': 0.5},
            {'x1': 2.0, 'x2': 2.2, 'y1': 3.5, 'y2': 4.5, 'f1': 0.6, 'f2': 0.6},
            {'x1': 1.0, 'x2': 1.2, 'y1': 3.0, 'y2': 4.0, 'f1': 0.7, 'f2': 0.7}
        ]
        self.db.insert_data(self.table_name, data)

        # Define conditions to select data where 'x1' is 1.0 and 'f1' is 0.5
        conditions = {'x1': 1.0, 'f1': 0.5}

        # Select data with conditions
        selected_data = self.db.select_data(self.table_name, conditions=conditions)
        print(selected_data)

        # Expected data that meets the conditions
        expected_data = [
            {'x1': 1.0, 'x2': 1.2, 'y1': 3.0, 'y2': 4.0, 'f1': 0.5, 'f2': 0.5}
        ]

        # Verify that the selected data matches the expected data
        self.assertDictListEqual(selected_data, expected_data)
     
    def test_09_delete_data(self):
        # Insert data
        data = {'x1': 0.5, 'x2': -0.5, 'y1': None, 'y2': None, 'f1': 0.1, 'f2': 0.1}
        self.db.insert_data(self.table_name, data)
    
        # Delete the data
        self.db.delete_data(self.table_name, 1)

        # Verify the deletion
        selected_data = self.db.select_data(self.table_name)
        self.assertEqual(len(selected_data), 0)


if __name__ == "__main__":
    unittest.main()
