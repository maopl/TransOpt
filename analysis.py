import logging
import os
import argparse
from ResultAnalysis.AnalysisPipeline import analysis_pipeline
from pathlib import Path

def run_analysis(Exper_folder:Path, tasks, methods, seeds):
    logger = logging.getLogger(__name__)
    analysis_pipeline(Exper_folder, tasks=tasks, methods=methods,seeds=seeds)



if __name__ == '__main__':
    tasks = {
        # 'cp': {'budget': 8, 'time_stamp': 2, 'params': {'input_dim': 2}},
        'Ackley': {'budget': 11, 'time_stamp': 3, 'params':{'input_dim':1}},
        # 'MPB': {'budget': 110, 'time_stamp': 3},
        # 'Griewank': {'budget': 11, 'time_stamp': 3,  'params':{'input_dim':1}},
        # 'DixonPrice': {'budget': 110, 'time_stamp': 3},
        # 'Lunar': {'budget': 110, 'time_stamp': 3},
        # 'XGB': {'budget': 110, 'time_stamp': 3},
    }
    Methods_list = {'MTBO', 'BO'}
    Seeds = [1,2,3,4,5]

    Exp_name = 'test'
    Exper_folder = '../LFL_experiments/{}'.format(Exp_name)
    Exper_folder = Path(Exper_folder)
    run_analysis(Exper_folder, tasks=tasks, methods=Methods_list,seeds= Seeds)

