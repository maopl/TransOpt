from .database import Database
from .lsh import LSHCache
from .minhash import MinHasher


class DataManager:
    def __init__(
        self, db=None, num_hashes=100, char_ngram=5, num_bands=50, random_state=12345
    ):
        if db is None:
            self.db = Database()
        else:
            self.db = db

        self._initialize_lsh_cache(num_hashes, char_ngram, num_bands, random_state)

    def _initialize_lsh_cache(self, num_hashes, char_ngram, num_bands, random_state):
        hasher = MinHasher(
            num_hashes=num_hashes, char_ngram=char_ngram, random_state=random_state
        )
        self.lsh_cache = LSHCache(hasher, num_bands=num_bands)

        datasets = self.db.get_table_list()

        for dataset in datasets:
            dataset_info = self.db.query_config(dataset)
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

    def get_similar_datasets(self, dataset_name, problem_config):
        vector = self._construct_vector(dataset_name, problem_config)
        similar_datasets = self.lsh_cache.query(vector)
        return similar_datasets

    def create_dataset(self, dataset_name, dataset_info, overwrite=True):
        self.db.create_table(dataset_name, dataset_info, overwrite)
        self._add_lsh_vector(dataset_name, dataset_info)

    def insert_data(self, dataset_name, data):
        return self.db.insert_data(dataset_name, data)
