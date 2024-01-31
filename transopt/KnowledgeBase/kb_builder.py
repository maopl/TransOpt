from pathlib import Path
from transopt.KnowledgeBase import KnowledgeBase
from transopt.KnowledgeBase.database import Database

def construct_knowledgebase(args):
    return Database()
    
    exp_folder = Path(args.exp_path) / args.exp_name
    file_path = exp_folder / args.optimizer / f'{args.seed}_KB.json'
    if hasattr(args, 'load_mode') and args.load_mode is not None:
        return KnowledgeBase(file_path, load_mode=args.load_mode)
    else:
        return KnowledgeBase(file_path, load_mode=False)