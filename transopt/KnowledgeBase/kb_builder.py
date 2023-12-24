from pathlib import Path
from transopt.KnowledgeBase import KnowledgeBase

def construct_knowledgebase(args):
    exp_folder = Path(args.exp_path) / args.exp_name
    file_path = exp_folder / args.optimizer / f'{args.seed}_KB.json'
    if hasattr(args, 'load_mode') or args.load_mode is not None:
        return KnowledgeBase(file_path, load_mode=args.load_mode)
    else:
        return KnowledgeBase(file_path, load_mode=False)