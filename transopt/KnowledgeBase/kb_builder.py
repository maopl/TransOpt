from pathlib import Path
from transopt.KnowledgeBase import KnowledgeBase

def construct_knowledgebase(args):
    exp_folder = Path(args.exp_path) / args.exp_name
    file_path = exp_folder / args.optimizer / f'{args.seed}_KB.json'
    return KnowledgeBase(file_path, load_mode=True)