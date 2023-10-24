import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pds
import os
import seaborn as sns
from Util.sk import Rx
from pathlib import Path
import scipy

table_registry = {}


# 注册函数的装饰器
def Tabel_register(name):
    def decorator(func_or_class):
        if name in Tabel_register:
            raise ValueError(f"Error: '{name}' is already registered.")
        Tabel_register[name] = func_or_class
        return func_or_class
    return decorator
