import numpy as np
from collections import defaultdict
from Util.sk import Rx
import scipy
from ResultAnalysis.AnalysisBase import AnalysisBase
table_registry = {}

# 注册函数的装饰器
def Tabel_register(name):
    def decorator(func_or_class):
        if name in table_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        table_registry[name] = func_or_class
        return func_or_class
    return decorator

@Tabel_register('best')
def record_best(ab:AnalysisBase, save_path, **kwargs):
    # Similar to print_best function in PeerComparison.py
    pass

@Tabel_register('mean')
def record_mean_std(ab:AnalysisBase, save_path, **kwargs):
    # Similar to record_mean_std function in PeerComparison.py
    pass

@Tabel_register('cr')
def record_acc_iterations(ab:AnalysisBase, save_path, **kwargs):
    # Similar to record_acc_iterations function in PeerComparison.py
    pass

@Tabel_register('convergence')
def record_convergence(ab:AnalysisBase, save_path, **kwargs):
    # Similar to record_convergence function in PeerComparison.py
    pass
