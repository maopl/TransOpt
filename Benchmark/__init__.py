import os
import pkgutil
import importlib

# 获取当前包的名字
package_name = __name__

# 准备一个字典来收集所有类
registry = {}

# 定义一个列表来包含要跳过的模块名或子包名
excluded_modules = ['hpobench', 'subpackage_to_skip', 'Remote']

# 定义一个函数用来注册类
def register_classes(module):
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        # 如果属性是一个类，并且是在这个模块中定义的（忽略导入的类）
        if isinstance(attribute, type) and attribute.__module__ == module.__name__:
            # 将类注册到registry字典中
            registry[attribute_name] = attribute

# 定义一个函数用来递归地导入所有模块和子包
def import_submodules(package, recursive=True):
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        # 跳过排除的模块或子包
        if name in excluded_modules:
            continue
        try:
            full_name = name
            results[full_name] = importlib.import_module(full_name)
            register_classes(results[full_name])
            if recursive and is_pkg:
                results.update(import_submodules(full_name))
        except ImportError as e:
            print(f"Failed to import {full_name}: {e}")
    return results

# 递归地导入当前包下的所有模块和子包
import_submodules(package_name)

# 将registry导出，以便可以从包外部访问
__all__ = ['registry']
