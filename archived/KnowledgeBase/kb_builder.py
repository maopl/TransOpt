from pathlib import Path
from transopt.KnowledgeBase.datamanager.database import Database

def construct_knowledgebase():
    # exp_folder = Path(args.exp_path) / args.exp_name
    # file_path = exp_folder / args.optimizer / f'{args.seed}_KB.json'
    return Database()
    # if hasattr(args, 'load_mode') and args.load_mode is not None:
    #     return Database()
    # else:
    #     return Database()