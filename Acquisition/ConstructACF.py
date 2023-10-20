
import GPyOpt

from Acquisition.EI import AcquisitionEI
from Acquisition.LCB import AcquisitionLCB
from Acquisition.ConformalLCB import ConformalLCB
from Acquisition.TAF import AcquisitionTAF_POE,AcquisitionTAF_M

from Util.Register import acf_registry


def get_ACF(acf_name, model, config, tabular=False):
    """Create the optimizer object."""
    acf_class = acf_registry.get(acf_name)

    if acf_class is not None:
        acquisition = acf_class(model=model, optimizer=config['optimizer'], config=config)
    else:
        # 处理任务名称不在注册表中的情况c
        print(f"Acquisition '{acf_name}' not found in the registry.")
        raise NameError
    if tabular:
        evaluator = Sequential(acquisition)
    else:
        evaluator = Sequential(acquisition)
    return evaluator, acquisition

