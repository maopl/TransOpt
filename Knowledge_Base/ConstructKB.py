from Knowledge_Base.KnowledgeBase import KnowledgeBase

def get_knowledgebase(args):
    Exper_folder = '{}/{}'.format(args.exp_path, args.exp_name)
    Method = args.optimizer
    Seed = args.seed
    KB = KnowledgeBase('{}/{}/{}_KB.json'.format(Exper_folder, Method, Seed))
    return KB