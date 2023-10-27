import numpy as np
from collections import defaultdict
from Util.sk import Rx
import scipy

table_registry = {}

def Tabel_register(name):
    def decorator(func_or_class):
        if name in table_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        table_registry[name] = func_or_class
        return func_or_class
    return decorator

@Tabel_register('record_best')
def record_best(results, save_path, **kwargs):
    # Similar to print_best function in PeerComparision.py
    pass

@Tabel_register('record_mean_std')
def record_mean_std(results, save_path, **kwargs):
    # Similar to record_mean_std function in PeerComparision.py
    pass

@Tabel_register('record_acc_iterations')
def record_acc_iterations(results, save_path, **kwargs):
    # Similar to record_acc_iterations function in PeerComparision.py
    pass

@Tabel_register('record_convergence')
def record_convergence(results, save_path, **kwargs):
    # Similar to record_convergence function in PeerComparision.py
    pass

def matrix_to_latex(mean, std, rst, col_names, row_names, oder='min'):
    # Similar to matrix_to_latex function in ToLatex.py
    pass