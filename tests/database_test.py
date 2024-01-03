import os
import sys
import unittest
from pathlib import Path

current_dir = Path(__file__).resolve().parent
package_dir = current_dir.parent
sys.path.insert(0, str(package_dir))

from transopt.KnowledgeBase.database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database("test_database.db")
        self.table_name = "test_table"
        dataset_info = {
            "variables": {"var1": {"type": "continuous"}, "var2": {"type": "integer"}},
            "objectives": ["obj1"],
        }

        # Table creation
        self.db.create_table(self.table_name, dataset_info)

    def tearDown(self):
        self.db.close()
        os.remove(self.db.data_path)

    def test_01_table_creation_and_removal(self):
        """Test the creation and removal of a table."""
        self.assertTrue(self.db.check_table_exist(self.table_name))

        # Test table removal
        self.db.remove_table(self.table_name)
        self.assertFalse(self.db.check_table_exist(self.table_name))

    def test_02_insert_data(self):
        """Test data insertion."""
        # Insert data
        data = [1.5, 2]  # Ensure this data matches your table schema
        self.db.insert_data(self.table_name, ["var1", "var2"], data)
        row_count = self.db.get_num_row(self.table_name)
        self.assertEqual(row_count, 1)  # Check that one row is inserted

    def test_03_insert_multiple_data(self):
        """Test the insertion of multiple data rows."""
        data_list = [
            [1.5, 2],
            [2.5, 3],
        ]  # List of lists, where each inner list represents a row

        # Insert multiple data rows
        self.db.insert_multiple_data(self.table_name, ["var1", "var2"], data_list)
        row_count = self.db.get_num_row(self.table_name)
        self.assertEqual(row_count, 2)
    
    def test_04_update_data(self):
        # Insert initial data
        initial_data = [1.5, 2]
        rowid = self.db.insert_data(self.table_name, ["var1", "var2"], initial_data)

        # Update the data
        new_data = [2.5, 3]
        self.db.update_data(self.table_name, ["var1", "var2"], new_data, rowid=rowid)

        # Verify the update
        updated_data = self.db.select_data(self.table_name, ["var1", "var2"], rowid=rowid)
        self.assertEqual(updated_data[0], new_data)


    def test_05_update_multiple_data(self):
        # Insert initial data
        initial_data = [
            [1.5, 2],
            [2.5, 3]
        ]
        rowids = self.db.insert_multiple_data(self.table_name, ["var1", "var2"], initial_data)

        # Update the data
        updated_data = [
            [3.5, 4],
            [4.5, 5]
        ]
        self.db.update_multiple_data(self.table_name, ["var1", "var2"], updated_data, [1, 2])

        # Verify the update
        new_data = self.db.select_data(self.table_name, ["var1", "var2"], rowid=rowids)
        self.assertEqual(new_data, updated_data)

    def test_06_delete_data(self):
        # Insert data
        self.db.insert_data(self.table_name, ["var1", "var2"], [1.5, 2])

        # Delete the data
        self.db.delete_data(self.table_name, 1)

        # Verify the deletion
        row_count = self.db.get_num_row(self.table_name)
        self.assertEqual(row_count, 0)
    

if __name__ == "__main__":
    unittest.main()
