
from Util.Register import optimizer_registry
from KnowledgeBase.KnowledgeBaseAccessor import KnowledgeBaseAccessor
import Optimizer


def get_optimizer(args):
    """Create the optimizer object."""
    optimizer_class = optimizer_registry.get(args.optimizer)
    config = {
        'init_method':args.init_method,
        'init_number':args.init_number,
        'normalize': args.normalize,
        'acf': args.acquisition_func,
        'verbose': args.verbose,
        'optimizer_name': args.optimizer,
        'save_path': args.exp_path,
    }

    if optimizer_class is not None:
        optimizer = optimizer_class(config=config)
    else:
        # 处理任务名称不在注册表中的情况
        print(f"Optimizer '{args.optimizer}' not found in the registry.")
        raise NameError
    return optimizer
