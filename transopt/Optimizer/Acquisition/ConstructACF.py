
import GPyOpt

from transopt.utils.Register import acf_registry


def get_ACF(acf_name, model, search_space, config, tabular=False):
    """Create the optimizer object."""
    acf_class = acf_registry.get(acf_name)
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(search_space)
    if acf_class is not None:
        acquisition = acf_class(model=model, optimizer=acquisition_optimizer, space=search_space, config=config)
    else:
        # 处理任务名称不在注册表中的情况c
        print(f"Acquisition '{acf_name}' not found in the registry.")
        raise NameError

    return acquisition

