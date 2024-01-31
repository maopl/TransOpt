import json
import sqlite3
import pandas as pd
from multiprocessing import Manager, Process, Queue

import numpy as np

from transopt.utils.path import get_library_path

'''
Descriptions of the reserved database tables.
'''
table_descriptions = {

    '_config': '''
        name varchar(50) not null,
        config text not null
        ''',

}


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

        # reserved tables
        self.reserved_tables = list(table_descriptions.keys())

        # reserved tables
        for name, desc in table_descriptions.items():
            if not self.check_table_exist(name):
                self.execute(f'CREATE TABLE "{name}" ({desc})')

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
        return [table[0] for table in table_list if table[0] not in self.reserved_tables]

    def check_table_exist(self, name):
        """Check if a certain database table exists."""
        table_exists = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            params=(name,),
            fetchone=True,
        )
        return table_exists is not None

    def create_table(self, name, problem_cfg, overwrite=False):
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
            if overwrite:
                self.remove_table(name)
            else:
                raise Exception(f"Table {name} already exists")

        variables = problem_cfg.get("variables", [])
        objectives = problem_cfg.get("objectives", [])
        fidelities = problem_cfg.get("fidelities", [])
    
        var_type_map = {
            "continuous": "float",
            "integer": "int",
            "categorical": "varchar(50)",
            # 'binary': 'boolean',
        }

        # description = ['status varchar(20) not null default "unevaluated"']
        description = []
        index_columns = []

        for var_info in variables:
            description.append(f'"{var_info["name"]}" {var_type_map[var_info["type"]]} not null')
            index_columns.append(var_info["name"])

        for obj_info in objectives:
            description.append(f'"{obj_info["name"]}" float')
        
        for fid_info in fidelities:
            description.append(f'"{fid_info["name"]}" {var_type_map[fid_info["type"]]} not null')
            index_columns.append(fid_info["name"]) 

        description += [
            "batch int default -1",
            "error boolean default 0",
            # "pareto boolean",
            # "batch int not null",
            # "order int default -1",
            # "hypervolume float",
        ]

        # Create the table
        self.execute(f'CREATE TABLE "{name}" ({",".join(description)})')

        # Create a combined index for variables and fidelity columns
        if index_columns:
            index_columns_string = ', '.join([f'"{column}"' for column in index_columns])
            self.execute(f'CREATE INDEX "idx_{name}_vars_fids" ON "{name}" ({index_columns_string})')
        
        self.create_or_update_config(name, problem_cfg) 

    def remove_table(self, name):
        if not self.check_table_exist(name):
            raise Exception(f"Table {name} does not exist")

        self.execute(f'DROP TABLE IF EXISTS "{name}"')

    '''
    config
    '''

    def create_or_update_config(self, name, problem_cfg):
        """
        Create or update a configuration entry in the _config table for a given table.
        """
        # Serialize problem_cfg into JSON format
        config_json = json.dumps(problem_cfg)

        # Check if the configuration already exists
        if self.query_config(name) is not None:
            # Update the existing configuration
            self.execute(
                "UPDATE _config SET config = ? WHERE name = ?",
                (config_json, name)
            )
        else:
            # Insert a new configuration
            self.execute(
                "INSERT INTO _config (name, config) VALUES (?, ?)",
                (name, config_json)
            )
        
    def query_config(self, name):
        config_json = self.execute(
            "SELECT config FROM _config WHERE name=?",
            params=(name,),
            fetchone=True
        )
        
        if config_json is None:
            return None
        else:
            return json.loads(config_json[0])

    def query_dataset_info(self, name):
        """
        Query the dataset information of a given table.
        """
        config = self.query_config(name)
   
        if config is None:
            return None

        variables = config["variables"]
        objectives = config["objectives"]
        fidelities = config["fidelities"]
        
        data_num = self.get_num_row(name)
        
        dataset_info = {
            "var_num": len(variables),
            "variables": variables,
            
            "obj_num": len(objectives),
            "objectives": objectives,
            
            "fidelities_num": len(fidelities),
            "fidelities": fidelities,
            
            "data_number": data_num,
        }
        return dataset_info 

    """
    basic operations
    """

    def commit(self):
        self.execute("COMMIT")

    def insert_data(self, table, data: dict or list or pd.DataFrame or np.ndarray) -> list:
        """
        Insert single-row or multiple-row data into the database.

        Parameters
        ----------
        table: str
            Name of the database table to insert into.
        data: dict, list, pd.DataFrame, or np.ndarray
            Data to insert. If a dictionary, it represents a single row of data
            where keys are column names and values are data values. If a list,
            each element represents a row (as a list or dict). If a DataFrame or
            np.ndarray, each row represents a row to be inserted.

        Returns
        -------
        list
            List of row numbers of the inse
        
        """
        if isinstance(data, dict):
            # Single row insertion from dict
            columns = list(data.keys())
            values = [list(data.values())]
        elif isinstance(data, list):
            # Multiple row insertion from list of dicts or lists
            if all(isinstance(row, dict) for row in data):
                columns = list(data[0].keys())
                values = [list(row.values()) for row in data]
            elif all(isinstance(row, list) for row in data):
                columns = None
                values = data
            else:
                raise ValueError("All rows in data_list must be of the same type (all dicts or all lists)")
        elif isinstance(data, (pd.DataFrame, np.ndarray)):
            # Convert DataFrame or ndarray to list of lists for insertion
            values = data.tolist() if isinstance(data, np.ndarray) else data.values.tolist()
            columns = data.columns.tolist() if isinstance(data, pd.DataFrame) else None
        else:
            raise ValueError("Data parameter must be a dictionary, list, pandas DataFrame, or numpy ndarray")

        if columns:
            column_str = ",".join([f'"{col}"' for col in columns])
            value_placeholders = ",".join(["?"] * len(columns))
        else:
            column_str = ""
            value_placeholders = ",".join(["?"] * len(values[0]))
        
        query = f'INSERT INTO "{table}" ({column_str}) VALUES ({value_placeholders})'
        self.executemany(query, values)
        self.commit()

        # Get the rowids of the inserted rows
        n_row = self.get_num_row(table)
        len_data = len(data) if isinstance(data, list) else len(values)
        return list(range(n_row - len_data + 1, n_row + 1))

    def _get_conditions(self, rowid=None, conditions=None):
        """
        Construct SQL conditions for a query based on rowid and additional conditions.

        Parameters
        ----------
        rowid: int/list
            Row number(s) of the table to query (if None then no rowid condition is added).
        conditions: dict
            Additional conditions for querying (key: column name, value: column value).

        Returns
        -------
        str
            SQL condition string.
        """
        from collections.abc import Iterable

        conditions_list = []

        # Handling rowid conditions
        if rowid is not None:
            if isinstance(rowid, Iterable) and not isinstance(rowid, str):
                rowid_condition = f'rowid IN ({",".join([str(r) for r in rowid])})'
                conditions_list.append(rowid_condition)
            else:
                conditions_list.append(f"rowid = {rowid}")

        # Handling additional conditions
        if conditions:
            for column, value in conditions.items():
                if isinstance(value, str):
                    value_str = f"'{value}'"  # Strings need to be quoted
                else:
                    value_str = str(value)
                condition_str = f'"{column}" = {value_str}'
                conditions_list.append(condition_str)

        # Combine all conditions with 'AND'
        if conditions_list:
            return " WHERE " + " AND ".join(conditions_list)
        else:
            return ""
    
    def update_data(self, table, data, rowid=None, conditions=None):
        """
        Update single-row or multiple-row data in the database.

        Parameters
        ----------
        table: str
            Name of the database table to update.
        data: dict or list of dicts
            Data to update. If a dictionary, it represents a single row of data
            where keys are column names and values are data values.
            If a list, each dictionary in the list represents a row to be updated.
        rowid: int/list
            Row number(s) of the table to update. If None, conditions are used.
        conditions: dict
            Additional conditions for updating (key: column name, value: column value).
        """
        if isinstance(data, dict):
            data = [data]

        update_values = []
        for row in data:
            columns = list(row.keys())
            values = list(row.values())
            set_clause = ", ".join([f'"{col}" = ?' for col in columns])
            query = f'UPDATE "{table}" SET {set_clause}'

            if rowid:
                query += f' WHERE rowid = ?'
                values.append(rowid)
            elif conditions:
                condition_str = " AND ".join([f'"{k}" = ?' for k in conditions.keys()])
                query += f' WHERE {condition_str}'
                values.extend(conditions.values())
            else:
                raise ValueError("Either rowid or conditions must be provided")

            update_values.append(values)

        self.executemany(query, update_values)
        self.commit()
    
    def delete_data(self, table, rowid=None, conditions=None):
        """
        Delete single-row or multiple-row data in the database.

        Parameters
        ----------
        table: str
            Name of the database table to delete from.
        rowid: int/list
            Row number(s) of the table to delete. If None, conditions are used.
        conditions: dict
            Additional conditions for deleting (key: column name, value: column value).
        """
        query = f'DELETE FROM "{table}"'
        condition = self._get_conditions(rowid=rowid, conditions=conditions)
        query += condition

        self.execute(query)
        self.commit()
    
    def select_data(self, table, columns=None, rowid=None, conditions=None, as_dataframe=False) -> list:
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
        conditions: dict
            Additional conditions for querying (key: column name, value: column value).
        as_dataframe: bool
            If True, return the result as a pandas DataFrame.

        Returns
        -------
        list of dicts
            Selected data, each row as a dictionary with column names as keys.
        """
        if columns is None:
            query = f'SELECT * FROM "{table}"'
            columns = self.get_column_names(table)
        elif isinstance(columns, str):
            query = f'SELECT "{columns}" FROM "{table}"'
            columns = [columns]
        else:
            column_str = ",".join([f'"{col}"' for col in columns])
            query = f'SELECT {column_str} FROM "{table}"'

        condition = self._get_conditions(rowid=rowid, conditions=conditions)
        query += condition

        # Convert each tuple in the results to a list
        results = self.execute(query, fetchall=True)
        if as_dataframe:
            return pd.DataFrame(results, columns=columns)
        else:
            return [dict(zip(columns, row)) for row in results]

    def get_num_row(self, table):
        query = f'SELECT COUNT(*) FROM "{table}"'
        return self.execute(query, fetchone=True)[0]
    
    def get_column_names(self, table):
        '''Get the column names of a database table. '''
        query = f'PRAGMA table_info("{table}")'
        return [col[1] for col in self.execute(query, fetchall=True)]
