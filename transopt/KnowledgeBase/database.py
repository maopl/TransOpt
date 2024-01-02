import os
import sqlite3
import sys
from collections.abc import Iterable
from multiprocessing import Event, Lock, Manager, Process, Queue

import numpy as np

from transopt.utils.path import get_library_path


def database_daemon(data_path, task_queue, result_queue, stop_event):
    with sqlite3.connect(data_path) as conn:
        cursor = conn.cursor()
        while not stop_event.is_set():
            task = task_queue.get()
            if task is None:  # Sentinel for stopping
                break

            func, args = task
            try:
                result = func(cursor, *args)
                result_queue.put(("SUCCESS", result))
            except Exception as e:
                result_queue.put(("FAILURE", e))


class Database:
    def __init__(self, db_file_name="database.db"):
        manager = Manager()
        self.data_path = get_library_path() / db_file_name

        self.task_queue = manager.Queue()
        self.result_queue = manager.Queue()
        self.stop_event = manager.Event()
        self.lock = manager.Lock()

        self.start_daemon()

    """
    connection
    """

    def start_daemon(self):
        self.process = Process(
            target=database_daemon,
            args=(self.data_path, self.task_queue, self.result_queue, self.stop_event),
        )
        self.process.start()

    def stop_daemon(self):
        self.stop_event.set()
        self.task_queue.put(None)
        self.process.join()

    def close(self):
        self.stop_daemon()

    """
    execution
    """

    def _execute(self, task, args=(), timeout=None):
        self.task_queue.put((task, args))
        try:
            # Queue.get() will block until the result is ready by default.
            status, result = self.result_queue.get(timeout=timeout)
            if status == "SUCCESS":
                return result
            else:
                raise result  # Re-raise the exception from the daemon
        except Queue.Empty:
            raise Exception("Task execution timed out or failed")

    @staticmethod
    def query_exec(cursor, query, params, fetchone, fetchall, many):
        if many:
            cursor.executemany(query, params or [])
        else:
            cursor.execute(query, params or ())

        if fetchone:
            return cursor.fetchone()
        if fetchall:
            return cursor.fetchall()
        return None

    def execute(self, query, params=None, fetchone=False, fetchall=False, timeout=None):
        with self.lock:
            return self._execute(
                Database.query_exec, (query, params, fetchone, fetchall, False), timeout
            )

    def executemany(
        self, query, params=None, fetchone=False, fetchall=False, timeout=None
    ):
        with self.lock:
            return self._execute(
                Database.query_exec, (query, params, fetchone, fetchall, True), timeout
            )

    """ 
    table
    """

    def get_table_list(self):
        """Get the list of all database tables."""
        table_list = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table'", fetchall=True
        )
        return [table[0] for table in table_list if table[0]]

    def check_table_exist(self, name):
        """Check if a certain database table exists."""
        table_exists = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            params=(name,),
            fetchone=True,
        )
        return table_exists is not None

    def create_table(self, name, dataset_info):
        """
        Create and initialize a database table based on problem configuration.

        Parameters
        ----------
        name: str
            Name of the table to create and initialize.
        problem_cfg: dict
            Configuration for the table schema.
        """
        if self.check_table_exist(name):
            raise Exception(f"Table {name} already exists")

        variables = dataset_info["variables"]
        objectives = dataset_info["objectives"]

        var_type_map = {
            "continuous": "float",
            "integer": "int",
            "categorical": "varchar(50)",
            # 'binary': 'boolean',
        }

        description = ['status varchar(20) not null default "unevaluated"']

        for var_name, var_info in variables.items():
            description.append(
                f'"{var_name}" {var_type_map[var_info["type"]]} not null'
            )

        for obj_name in objectives:
            description.append(f'"{obj_name}" float')

        description += [
            "pareto boolean",
            # "batch int not null",
            # "order int default -1",
            "hypervolume float",
        ]

        self.execute(f'CREATE TABLE "{name}" ({",".join(description)})')

    def remove_table(self, name):
        if not self.check_table_exist(name):
            raise Exception(f"Table {name} does not exist")

        self.execute(f'DROP TABLE IF EXISTS "{name}"')

    """
    basic operations
    """

    def commit(self):
        self.execute("COMMIT")

    def insert_data(self, table, columns, data):
        """
        Insert single-row data to the database.

        Parameters
        ----------
        table: str
            Name of the database table to insert.
        column: str/list
            Column name(s) of the table to insert.
        data: list/np.ndarray
            Data to insert.

        Returns
        -------
        int
            Row number of the inserted data.
        """

        if type(data) == np.ndarray:
            data = data.tolist()

        if columns is None:
            query = f'INSERT INTO "{table}" VALUES ({",".join(["?"] * len(data))})'
        elif type(columns) == str:
            query = f'INSERT INTO "{table}" ("{columns}") VALUES (?)'
        elif isinstance(columns, list):
            column_str = ",".join([f'"{col}"' for col in columns])
            query = f'INSERT INTO "{table}" ({column_str}) VALUES ({",".join(["?"] * len(data))})'
        else:
            raise ValueError("Column parameter must be a string or list of strings")

        self.execute(query, data)
        self.commit()  # Ensure the data is committed if not auto-committed

        rowid = self.get_num_row(table)
        return rowid

    def insert_multiple_data(self, table, columns, data_list):
        """
        Insert multiple rows of data to the database.

        Parameters
        ----------
        table : str
            Name of the database table to insert.
        columns : list
            Column names of the table to insert.
        data_list : list of lists/np.ndarray
            Data to insert, each inner list is a row.

        Returns
        -------
        list
            Row numbers of the inserted data.
        """
        if type(data_list) == np.ndarray:
            data_list = data_list.tolist()

        if columns is None:
            query = (
                f'INSERT INTO "{table}" VALUES ({",".join(["?"] * len(data_list[0]))})'
            )
        elif type(columns) == str:
            query = f'INSERT INTO "{table}" ("{columns}") VALUES (?)'
        elif isinstance(columns, list):
            column_str = ",".join([f'"{col}"' for col in columns])
            query = f'INSERT INTO "{table}" ({column_str}) VALUES ({",".join(["?"] * len(data_list[0]))})'
        else:
            raise ValueError("Column parameter must be a string or list of strings")

        self.executemany(query, data_list)
        self.commit()

        n_row = self.get_num_row(table)
        return list(range(n_row - len(data_list) + 1, n_row + 1))

    def _get_rowid_condition(self, rowid):
        if rowid is None:
            return ""
        elif isinstance(rowid, Iterable):
            return f' WHERE rowid IN ({",".join([str(r) for r in rowid])})'
        else:
            return f" WHERE rowid = {rowid}"

    def update_data(self, table, columns, data, rowid: int):
        """
        Update single-row data in the database.

        Parameters
        ----------
        table: str
            Name of the database table to update.
        columns: str/list
            Column name(s) of the table to update.
        data: list/np.ndarray
            Data to update.
        rowid: int
            Row number of the table to update.
        """
        if type(data) == np.ndarray:
            data = data.tolist()

        if type(columns) == str:
            query = f'UPDATE "{table}" SET "{columns}" = ?'
        else:
            column_str = ",".join([f'"{col}" = ?' for col in columns])
            query = f'UPDATE "{table}" SET {column_str}'

        condition = self._get_rowid_condition(rowid)
        query += condition

        self.execute(query, data)
        self.commit()

    def update_multiple_data(self, table, columns, data_list, rowid_list):
        """
        Update multiple rows of data in the database.

        Parameters
        ----------
        table : str
            Name of the database table to update.
        columns : str/list
            Column names of the table to update.
        data_list : list of lists/np.ndarray
            Data to update, each inner list is a row.
        rowid_list : list of ints
            Row numbers of the table to update.
        """
        if type(data_list) == np.ndarray:
            data_list = data_list.tolist()

        if len(rowid_list) != len(data_list):
            raise ValueError("rowid_list must be provided and match the length of data_list.")

        if type(columns) == str:
            query = f'UPDATE "{table}" SET "{columns}" = ? WHERE rowid = ?'
        else:
            column_str = ",".join([f'"{col}" = ?' for col in columns])
            query = f'UPDATE "{table}" SET {column_str} WHERE rowid = ?'

        combined_data = [data + [rowid] for data, rowid in zip(data_list, rowid_list)]

        self.executemany(query, combined_data)
        self.commit()

    def delete_data(self, table, rowid: int):
        """
        Delete single-row data in the database.

        Parameters
        ----------
        table: str
            Name of the database table to delete.
        rowid: int
            Row number of the table to delete.
        """
        query = f'DELETE FROM "{table}"'
        condition = self._get_rowid_condition(rowid)
        query += condition

        self.execute(query)
        self.commit()

    def select_data(self, table, columns=None, rowid=None):
        """
        Select data in the database.

        Parameters
        ----------
        table: str
            Name of the database table to query.
        column: str/list
            Column name(s) of the table to query (if None then select all columns).
        rowid: int/list
            Row number(s) of the table to query (if None then select all rows).

        Returns
        -------
        list
            Selected data.
        """
        if columns is None:
            query = f'SELECT * FROM "{table}"'
        elif type(columns) == str:
            query = f'SELECT "{columns}" FROM "{table}"'
        else:
            column_str = ",".join([f'"{col}"' for col in columns])
            query = f'SELECT {column_str} FROM "{table}"'

        condition = self._get_rowid_condition(rowid)
        query += condition

        # Convert each tuple in the results to a list
        results = self.execute(query, fetchall=True)
        results = [list(row) for row in results]

        return results

    def get_num_row(self, table):
        query = f'SELECT COUNT(*) FROM "{table}"'
        return self.execute(query, fetchone=True)[0]
    
    def get_column_names(self, table):
        '''Get the column names of a database table. '''
        query = f'PRAGMA table_info("{table}")'
        return [col[1] for col in self.execute(query, fetchall=True)]
