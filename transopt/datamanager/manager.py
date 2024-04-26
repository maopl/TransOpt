import cProfile
import pstats

from transopt.datamanager.database import Database
from transopt.datamanager.lsh import LSHCache
from transopt.datamanager.minhash import MinHasher


class DataManager:
    _instance = None
    _init = False  # 用于保证初始化代码只运行一次

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db=None, num_hashes=100, char_ngram=5, num_bands=50, random_state=12345):
        if not self._initialized:
            if db is None:
                self.db = Database()
            else:
                self.db = db

            self._initialize_lsh_cache(num_hashes, char_ngram, num_bands, random_state)
            self._initialized = True

    def _initialize_lsh_cache(self, num_hashes, char_ngram, num_bands, random_state):
        hasher = MinHasher(
            num_hashes=num_hashes, char_ngram=char_ngram, random_state=random_state
        )
        self.lsh_cache = LSHCache(hasher, num_bands=num_bands)

        datasets = self.db.get_table_list()

        for dataset in datasets:
            dataset_info = self.db.query_dataset_info(dataset)
            self._add_lsh_vector(dataset, dataset_info)

    def _add_lsh_vector(self, dataset_name, dataset_info):
        vector = self._construct_vector(dataset_name, dataset_info)
        self.lsh_cache.add(dataset_name, vector)

    def _construct_vector(self, dataset_name, dataset_info):
        num_variables = dataset_info["num_variables"]
        num_objectives = dataset_info["num_objectives"]

        variables = dataset_info["variables"]
        variable_names = " ".join([var["name"] for var in variables])

        return (dataset_name, variable_names, num_variables, num_objectives)

    def search_similar_datasets(self, dataset_name, problem_config):
        vector = self._construct_vector(dataset_name, problem_config)
        similar_datasets = self.lsh_cache.query(vector)
        return similar_datasets

    def search_datasets_by_name(self, dataset_name):
        all_tables = self.db.get_table_list()
        matching_tables = [table for table in all_tables if dataset_name.lower() in table.lower()]
        return matching_tables
    
    def get_dataset_info(self, dataset_name):
        return self.db.query_dataset_info(dataset_name)
    
    def get_all_datasets(self):
        return self.db.get_table_list()
        
    def create_dataset(self, dataset_name, dataset_info, overwrite=True):
        self.db.create_table(dataset_name, dataset_info, overwrite)
        self._add_lsh_vector(dataset_name, dataset_info)

    def insert_data(self, dataset_name, data):
        return self.db.insert_data(dataset_name, data)

    def teardown(self):
        self.db.close()


def main():
    dm = DataManager(num_hashes=200, char_ngram=5, num_bands=100)

    dataset = dm.db.get_table_list()[0]
    test_query = dm.db.query_dataset_info(dataset)
    
    sd = dm.search_similar_datasets(dataset, test_query)
    
    print(dm.db.get_table_list()[:2])

    dm.teardown()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.run('main()')
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('time').print_stats(10)