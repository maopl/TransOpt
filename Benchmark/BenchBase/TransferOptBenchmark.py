import abc
from typing import Union, Dict, List
import ConfigSpace
import logging
import numpy as np
from . import ConfigOptBenchmark
from KnowledgeBase.TaskDataHandler import OptTaskDataHandler
import matplotlib.pyplot as plt
logger = logging.getLogger('TransferOptBenchmark')


class TransferOptBenchmark(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, seed: Union[int, np.random.RandomState, None] = None, **kwargs):
        self.seed = seed
        self.tasks = []
        self.query_nums = []
        self.__id = 0

    def add_task_to_id(self, insert_id:int,
                       task:Union[ConfigOptBenchmark.NonTabularOptBenchmark, ConfigOptBenchmark.TabularOptBenchmark]):
        num_tasks = len(self.tasks)
        assert insert_id < num_tasks + 1

        self.tasks.insert(insert_id, task)
        self.query_nums.insert(insert_id, 0)

    def add_task(self, task:Union[ConfigOptBenchmark.NonTabularOptBenchmark, ConfigOptBenchmark.TabularOptBenchmark]):
        num_tasks = len(self.tasks)
        insert_id = num_tasks
        self.add_task_to_id(insert_id, task)

    def del_task_by_id(self, del_id, name):
        pass

    def get_curid(self):
        return self.__id

    def get_tasks_num(self):
        return len(self.tasks)

    def get_unsolved_num(self):
        return len(self.tasks) - self.__id

    def get_rest_budget(self):
        return self.get_curbudget() - self.get_query_num()

    def get_query_num(self):
        return self.query_nums[self.__id]

    def get_curbudget(self):
        return self.tasks[self.__id].get_budget()

    def get_curname(self):
        return self.tasks[self.__id].get_name()

    def get_curdim(self):
        return self.tasks[self.__id].get_input_dim()

    def get_curobjnum(self):
        return self.tasks[self.__id].get_objective_num()
    def get_curtask(self):
        return self.tasks[self.__id]

    def get_curcs(self):
        return self.tasks[self.__id].configuration_space

    def get_curseed(self):
        return self.tasks[self.__id].seed

    def get_curtask_id(self):
        return self.tasks[self.__id].task_id

    def sync_query_num(self, query_num: int):
        self.query_nums[self.__id] = query_num

    def roll(self):
        self.__id += 1

    def lock(self):
        self.tasks[self.__id].lock()

    def unlock(self):
        self.tasks[self.__id].unlock()

    def get_lockstate(self):
        return self.tasks[self.__id].get_lock_state()

    def get_task_type(self):
        if isinstance(self.tasks[self.__id], ConfigOptBenchmark.TabularOptBenchmark):
            return 'tabular'
        elif isinstance(self.tasks[self.__id], ConfigOptBenchmark.NonTabularOptBenchmark):
            return 'Continuous'
        else:
            logger.error('Unknown task type.')
            raise NameError

    def get_cur_space_info(self)->Dict:
        space_info = {'input_dim': self.get_curdim(), 'num_objective': self.get_curobjnum(), 'budget': self.get_curbudget(), 'seed': self.get_curseed(), 'task_id':self.get_curtask_id()}
        cs = self.get_curcs()

        for k, v in cs.items():
            if type(v) is ConfigSpace.CategoricalHyperparameter:
                space_info[k] = {'bounds': [0, len(v.choices) - 1], 'type': type(v).__name__}
            else:
                space_info[k] = {'bounds':[v.lower, v.upper], 'type':type(v).__name__}

        return space_info

    ###Methods only for tabular data###
    def get_dataset_size(self):
        assert isinstance(self.tasks[self.__id], ConfigOptBenchmark.TabularOptBenchmark)
        return  self.tasks[self.__id].get_dataset_size()

    def get_var_by_idx(self, idx):
        assert isinstance(self.tasks[self.__id], ConfigOptBenchmark.TabularOptBenchmark)
        return self.tasks[self.__id].get_var_by_idx(idx)

    def get_idx_by_var(self, vectors):
        assert isinstance(self.tasks[self.__id], ConfigOptBenchmark.TabularOptBenchmark)
        return self.tasks[self.__id].get_idx_by_var(vectors)

    def get_unobserved_vars(self):
        assert isinstance(self.tasks[self.__id], ConfigOptBenchmark.TabularOptBenchmark)
        return self.tasks[self.__id].get_unobserved_vars()

    def get_unobserved_idxs(self):
        assert isinstance(self.tasks[self.__id], ConfigOptBenchmark.TabularOptBenchmark)
        return self.tasks[self.__id].get_unobserved_idxs()

    def add_query_num(self):
        if self.get_lockstate() == False:
            self.query_nums[self.__id] += 1



    def f(self, configuration: Union[ConfigSpace.Configuration, Dict, List[Union[ConfigSpace.Configuration, Dict]]],
                           fidelity: Union[Dict, ConfigSpace.Configuration, None, List[Union[ConfigSpace.Configuration, Dict]]] = None,
                           idx: Union[int, None, List[int]] = None,
                           **kwargs):
        if isinstance(configuration, list):
            if self.get_query_num() + len(configuration) > self.get_curbudget() and self.get_lockstate() == False:
                logger.error(' The current function evaluation has exceeded the user-set budget.')
                raise EnvironmentError

            if isinstance(fidelity, list):
                assert len(fidelity) == len(configuration)
            elif fidelity is None:
                fidelity = [None] * len(configuration)
            else:
                pass

            if isinstance(idx, list):
                assert len(idx) == len(configuration)

            results = []
            for c_id, config in enumerate(configuration):
                if isinstance(self.tasks[self.__id], ConfigOptBenchmark.TabularOptBenchmark):
                    result = self.tasks[self.__id].f(config, fidelity[c_id], idx[c_id])

                elif isinstance(self.tasks[self.__id], ConfigOptBenchmark.NonTabularOptBenchmark):
                    result = self.tasks[self.__id].f(config, fidelity[c_id])
                else:
                    raise TypeError(f'Unrecognized task type.')

                self.add_query_num()

                results.append(result)
            return results
        else:
            if self.get_query_num() >= self.get_curbudget() and self.get_lockstate() == False:
                logger.error(' The current function evaluation has exceeded the user-set budget.')
                raise EnvironmentError

            if isinstance(self.tasks[self.__id], ConfigOptBenchmark.TabularOptBenchmark):
                return self.tasks[self.__id].f(configuration, fidelity, idx)

            if isinstance(self.tasks[self.__id], ConfigOptBenchmark.NonTabularOptBenchmark):
                return self.tasks[self.__id].f(configuration, fidelity)

            self.add_query_num()

            raise TypeError(f'Unrecognized task type.')