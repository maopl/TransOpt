import logging
import os
import argparse
from ResultAnalysis.AnalysisPipeline import analysis_pipeline
from pathlib import Path

def run_analysis(tasks, Exper_folder:Path, Methods_list, Seeds):
    logger = logging.getLogger(__name__)
    analysis_pipeline(Exper_folder, Methods_list, Seeds)



if __name__ == '__main__':
    if __name__ == '__main__':
        tasks = {
            'Ackley': {'budget': 11, 'time_stamp': 3, 'params': {'input_dim': 1}},
            # 'MPB': {'budget': 110, 'time_stamp': 3},
            'Griewank': {'budget': 11, 'time_stamp': 3, 'params': {'input_dim': 1}},
            # 'DixonPrice': {'budget': 110, 'time_stamp': 3},
            # 'Lunar': {'budget': 110, 'time_stamp': 3},
            # 'XGB': {'budget': 110, 'time_stamp': 3},
        }
        Methods_list = {'MTBO', 'BO'}
        Seeds = [1]

        Exp_name = 'test'
        Exper_folder = '../LFL_experiments/{}'.format(Exp_name)
        Exper_folder = Path(Exper_folder)
        run_analysis(tasks, Exper_folder, Methods_list, Seeds)

