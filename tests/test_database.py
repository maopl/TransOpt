import os
import sys
import unittest
from pathlib import Path

current_dir = Path(__file__).resolve().parent
package_dir = current_dir.parent
sys.path.insert(0, str(package_dir))

import numpy as np
import pandas as pd
from transopt.datamanager.database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database("test_database.db")
        self.table_name = "test_table"
        self.dataset_info = {
            "variables": [
                {"name": "x1", "type": "continuous", "lb": -5.12, "ub": 5.12, "default": 0.0},
                {"name": "x2", "type": "categorical", "categories": ['1', '2', '3'], "default": 2},
                {"name": "x3", "type": "log_continuous", "lb": 0.1, "ub": 10, "base": 10, "default": 1.0},    
                {"name": "x4", "type": "int_exponent", "lb": 1, "ub": 512, "base": 2, "default": 1},
                {"name": "x5", "type": "integer", "lb": 1, "ub": 10, "default": 5},
            ],
            
            "objectives": [
                {"name": "y1", "type": "minimize"},
                {"name": "y2", "type": "maximize"},
            ],

            "fidelities": [
                {"name": "f1", "type": "continuous", "range": [0, 1], "default": 0.0},
                {"name": "f2", "type": "continuous", "range": [0, 1], "default": 0.0},
            ]
        }

        # Table creation
        self.db.create_table(self.table_name, self.dataset_info, overwrite=True)

    def tearDown(self):
        self.db.close()
        os.remove(self.db.data_path)
    
    def assertDictListEqual(self, list1, list2, msg=None):
        """Helper method to assert that two lists of dictionaries are equal."""
        self.assertEqual(len(list1), len(list2), msg)
        for dict1, dict2 in zip(list1, list2):
            for key in dict1:
                self.assertEqual(dict1[key], dict2[key], msg)
                
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
        # Adjust the row data to match the variable types and names in dataset_info
        row = {'x1': -2.0, 'x2': '2', 'x3': np.log10(5), 'x4': 16, 'x5': 5, 'y1': 1.5, 'y2': 2.5, 'f1': 0.5, 'f2': 0.5}
        rowid = self.db.insert_data(self.table_name, row)[0]
        selected_data = self.db.select_data(self.table_name, rowid=rowid, as_dataframe=True)
        
        filtered_selected_data = selected_data[list(row.keys())]
        expected_data = pd.DataFrame([row])
        pd.testing.assert_frame_equal(filtered_selected_data.reset_index(drop=True), expected_data.reset_index(drop=True), check_dtype=False)
       
    
    def test_04_insert_multiple_rows_list(self):
        """Test inserting and selecting multiple rows with a list."""
        # Adjust the rows data to match the variable types and names in dataset_info
        rows = [
            {'x1': -2.0, 'x2': '2', 'x3': np.log10(5), 'x4': 16, 'x5': 5, 'y1': 1.5, 'y2': 2.5, 'f1': 0.5, 'f2': 0.5},
            {'x1': 1.0, 'x2': '3', 'x3': np.log10(2), 'x4': 4, 'x5': 2, 'y1': 2.5, 'y2': 3.5, 'f1': 0.6, 'f2': 0.6}
        ]
        rowids = self.db.insert_data(self.table_name, rows)
        selected_data = [self.db.select_data(self.table_name, rowid=rowid, as_dataframe=True).drop(columns=['batch', 'error']) for rowid in rowids]
        expected_data = pd.DataFrame(rows)
        for i, df in enumerate(selected_data):
            pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_data.iloc[[i]].reset_index(drop=True), check_dtype=False)
             
    def test_05_insert_with_dataframe(self):
        """Test inserting and selecting data with a pandas DataFrame."""
        # Adjust the DataFrame to match the variable types and names in dataset_info
        df = pd.DataFrame({
            'x1': [-2.0, 1.0],
            'x2': ['2', '3'],
            'x3': np.log10([5, 2]),
            'x4': [16, 4],
            'x5': [5, 2],
            'y1': [1.5, 2.5],
            'y2': [2.5, 3.5],
            'f1': [0.5, 0.6],
            'f2': [0.5, 0.6]
        })
        rowids = self.db.insert_data(self.table_name, df)
        selected_data = [self.db.select_data(self.table_name, rowid=rowid, as_dataframe=True)[df.columns] for rowid in rowids]
        for i, df_selected in enumerate(selected_data):
            pd.testing.assert_frame_equal(df_selected.reset_index(drop=True), df.iloc[[i]].reset_index(drop=True), check_dtype=False)

    def test_06_insert_with_ndarray(self):
        """Test inserting and selecting data with a numpy ndarray."""
        # Adjust the ndarray to match the variable types and names in dataset_info
        arr = np.array([
            [-2.0, '2', np.log10(5), 16, 5, 1.5, 2.5, 0.5, 0.5],
            [1.0, '3', np.log10(2), 4, 2, 2.5, 3.5, 0.6, 0.6]
        ])
        columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y1', 'y2', 'f1', 'f2']
        dtypes = {'x1': float, 'x2': str, 'x3': float, 'x4': int, 'x5': int, 'y1': float, 'y2': float, 'f1': float, 'f2': float}
        df = pd.DataFrame(arr, columns=columns).astype(dtypes)
        rowids = self.db.insert_data(self.table_name, df)
        selected_data = [self.db.select_data(self.table_name, rowid=rowid, as_dataframe=True).drop(columns=['batch', 'error']) for rowid in rowids]
        
        print(selected_data)
        for i, df_selected in enumerate(selected_data):
            pd.testing.assert_frame_equal(df_selected.reset_index(drop=True), df.iloc[[i]].reset_index(drop=True))
             
    def test_07_select_data_with_conditions(self):
        """Test selecting data with conditions."""
        # Insert example data
        data = [
            {'x1': 1.0, 'x2': '1', 'x3': np.log10(2), 'x4': 2, 'x5': 3, 'y1': None, 'y2': None, 'f1': 0.5, 'f2': 0.5},
            {'x1': 2.0, 'x2': '2', 'x3': np.log10(3), 'x4': 4, 'x5': 5, 'y1': None, 'y2': None, 'f1': 0.6, 'f2': 0.6}
        ]
        self.db.insert_data(self.table_name, data)

        # Conditions to select data
        conditions = {'x1': 1.0, 'x2': '1'}

        # Select data with conditions
        selected_data = self.db.select_data(self.table_name, conditions=conditions, as_dataframe=True).drop(columns=['batch', 'error'])
        print(selected_data)

        # Expected data that meets the conditions
        expected_data = pd.DataFrame([{
            'x1': 1.0, 'x2': '1', 'x3': np.log10(2), 'x4': 2, 'x5': 3, 'y1': None, 'y2': None, 'f1': 0.5, 'f2': 0.5
        }])
        
        # Verify that the selected data matches the expected data
        pd.testing.assert_frame_equal(selected_data.reset_index(drop=True), expected_data.reset_index(drop=True), check_dtype=False)

 
    # def test_09_delete_data(self):
    #     # Insert data
    #     data = {'x1': 0.5, 'x2': -0.5, 'y1': None, 'y2': None, 'f1': 0.1, 'f2': 0.1}
    #     self.db.insert_data(self.table_name, data)
    
    #     # Delete the data
    #     self.db.delete_data(self.table_name, 1)

    #     # Verify the deletion
    #     selected_data = self.db.select_data(self.table_name)
    #     self.assertEqual(len(selected_data), 0)


if __name__ == "__main__":
    unittest.main()
