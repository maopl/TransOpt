import os
import pkgutil
import importlib

# 基础包名
base_package_name = __name__

# 准备一个字典来收集所有类
registry = {}

# 定义一个函数用来注册类
def register_classes(module):
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        # 如果属性是一个类，并且是在这个模块中定义的
        if isinstance(attribute, type) and attribute.__module__ == module.__name__:
            # 将类注册到registry字典中
            registry[attribute_name] = attribute

# 定义一个函数用来导入指定文件夹下的所有模块
def import_modules_from_dir(directory, folder_name):
    for (finder, name, ispkg) in pkgutil.iter_modules([directory]):
        if not ispkg:
            module_name = f"{base_package_name}.{folder_name}.{name}"
            module = importlib.import_module(module_name)
            register_classes(module)

# 递归地导入 Optimizer 下的 singleOptimizer 和 MultiOptimizer 子文件夹中的模块
subfolders = ['SingleObjOptimizer', 'MultiObjOptimizer']
for subfolder in subfolders:
    subfolder_path = os.path.join(os.path.dirname(__file__), subfolder)
    import_modules_from_dir(subfolder_path, subfolder)

# 将registry导出，以便可以从包外部访问
__all__ = ['registry']
