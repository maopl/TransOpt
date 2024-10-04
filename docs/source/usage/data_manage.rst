Data Management
===============


Design Overview
---------------

The `datamanager` module is designed to handle optimization task data using a combination of an SQLite-based database and a Locality-Sensitive Hashing (LSH) mechanism for similarity search. The module’s core class, `DataManager`, manages the interaction with the database and the LSH cache, providing a unified interface for storing, retrieving, and searching datasets.

- **Database Submodule**: 
  Uses an SQLite database to store datasets, their configurations, and metadata. It provides methods for creating tables, inserting data, querying, and managing transaction operations.

- **LSH Submodule**: 
  Implements an LSH mechanism using MinHash to facilitate efficient similarity searches, storing MinHash fingerprints of datasets for quick comparison.

Data Flow
---------

**Initialization**:

- The `DataManager` class is a singleton that initializes the database (using the `Database` submodule) and the LSH cache. 
- During initialization, `DataManager` sets up the LSH cache by fetching datasets from the database and constructing MinHash vectors for indexing.

**Data Input**:

- New datasets are created in the database using the `create_dataset` method, which utilizes the `Database` submodule to define the table schema and insert metadata. 
- The `_add_lsh_vector` method in `DataManager` then computes an LSH vector and adds it to the LSH cache.
- Data can be inserted into existing tables via the `insert_data` method of `DataManager`, which interacts with the `Database` submodule.

**Data Storage**:

- **Database Submodule**:

  The `Database` submodule uses an SQLite database for data storage. It includes methods to create tables, insert data, manage configurations, and store metadata related to datasets. It operates through a `DatabaseDaemon` process, managing database operations using a task queue and ensuring that all database interactions are thread-safe. Predefined tables such as `_config` and `_metadata` store configuration and metadata for datasets, allowing `DataManager` to query and manage them effectively.

- **LSH Submodule**:

  The `LSHCache` class stores MinHash fingerprints of datasets, facilitating efficient similarity searches.

**Data Processing**:

- **MinHashing and LSH**:

  The `MinHasher` class processes dataset information, generating MinHash signatures based on textual attributes (e.g., problem name, variable names). It uses character n-grams and a set number of hash functions to create a fingerprint of the dataset. The `LSHCache` class divides these fingerprints into bands, storing them in buckets for quick similarity comparisons.

- **Database Operations**:

  The `Database` submodule provides methods to handle complex operations such as table creation (`create_table`), dataset querying (`get_all_datasets`, `get_experiment_datasets`), and transaction management (`start_transaction`, `commit_transaction`, `rollback_transaction`). The `DatabaseDaemon` ensures that operations are executed sequentially and safely using a separate process, communicating results through task and result queues.

**Data Retrieval**:

- **Direct Queries**:

  `get_dataset_info` and `get_all_datasets` in `DataManager` interact with the `Database` submodule to fetch detailed information and lists of datasets.

- **Similarity Search**:

  `search_similar_datasets` in `DataManager` uses the LSH cache to find datasets similar to a given problem configuration based on MinHash signatures.

**Data Modification**:

- The `create_dataset` method in `DataManager` creates a new table in the database, setting up its schema based on a configuration dictionary. The method optionally adds indices for specified columns to optimize query performance.
- The `remove_dataset` method removes tables and associated data from both the database and the LSH cache.

**Teardown**:

- The `teardown` method resets the `DataManager` instance and ensures proper closure of the database by signaling the `DatabaseDaemon` to terminate.

Interaction Between Submodules
------------------------------

- **Database Submodule**:

  Manages all direct interactions with the SQLite database for data storage, retrieval, and modification. It initializes essential tables (`_config`, `_metadata`) during its setup if they do not exist. Uses transaction locks and a daemon process to handle database operations asynchronously, ensuring data integrity and avoiding deadlocks.

- **LSH Submodule**:

  The `DataManager` class initializes an `LSHCache` using a `MinHasher` instance to compute and store MinHash fingerprints of datasets. The `LSHCache` enables fast similarity search by organizing MinHash vectors into buckets, allowing for efficient querying and comparison of datasets based on key attributes.

Submodule Details
-----------------

- **Database Submodule**:

  - **DatabaseDaemon**:

    Runs in a separate process to manage SQLite database interactions using task and result queues. Handles execution of database queries, managing transactions and error handling.

  - **Key Methods**:

    - **create_table**: Creates a new table in the database based on a provided configuration, including columns for variables, objectives, and fidelities.
    
    - **insert_data**: Inserts new data entries into a specified table.
    
    - **get_experiment_datasets**: Retrieves a list of tables marked as experiment datasets.
    
    - **Transaction Management**: Supports starting, committing, and rolling back transactions for batch operations.

- **LSH Submodule**:
  
  - **MinHasher**:

    Generates MinHash signatures for text inputs using character n-grams and multiple hash functions.

    - **fingerprint**: Computes a MinHash fingerprint for a given text input.

  - **LSHCache**:

    Divides MinHash signatures into bands for efficient similarity comparison.

    - **add**: Stores a dataset’s vector in the LSH cache.
    - **query**: Searches for similar vectors in the cache, returning keys of matching datasets.

This comprehensive architecture allows the `DataManager` to manage optimization datasets effectively, supporting both direct retrieval and similarity-based searching through a combination of database operations and the LSH mechanism.
