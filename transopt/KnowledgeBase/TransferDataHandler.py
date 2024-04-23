
from typing import List, Any, Union, Tuple, Dict, Callable

import archived.DataSelection
from transopt.KnowledgeBase.datamanager.database import Database
from transopt.KnowledgeBase.DataHandlerBase import DataHandler

class TransferDataHandler(DataHandler):

    def __init__(self, db: Database, args):
        super(TransferDataHandler, self).__init__(db, args)


    def add_observation(self, input_vectors: Union[List[Dict], Dict], output_value: Union[List[Dict], Dict]) -> None:
        """
        Adds new observations to the dataset.

        Args:
        - input_vectors (Union[List[Dict], Dict]): A list of dictionaries or a single dictionary, each containing input vector data.
        - output_value (Union[List[Dict], Dict]): A list of dictionaries or a single dictionary containing output values and associated info.
        """
        # Normalize input_vectors and output_value to lists for consistency
        if not isinstance(input_vectors, list):
            input_vectors = [input_vectors]

        if not isinstance(output_value, list):
            output_value = [output_value]

        # Check if the lengths of input_vectors and output_value match
        if len(input_vectors) != len(output_value):
            raise ValueError("The number of input vectors must match the number of output values.")

        # Validate each input vector
        for iv in input_vectors:
            self._validate_input_vector(iv)

        for ov in output_value:
            self._validate_output_value(ov)

        # Add to dataset
        self.dataset['input_vector'].extend(input_vectors)
        self.dataset['output_value'].extend(output_value)

        self._flush_dataset()

    def _flush_dataset(self):
        self.db.update_dataset(self.dataset_id, self.dataset['name'], self.dataset)
        self.db._save_database()

    def data_augmentation(self) -> Dict:
        return {}
    def metadata_selection(self) -> Dict:
        return {}

    def get_auxillary_data(self):
        if self.selector is None:
            return {}
        return  self.selector(self, self.args)



