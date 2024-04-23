
# 定义一个空的注册表字典
optimizer_registry = {}
problem_registry = {}
normalizer_registry = {}
acf_registry = {}
para_registry = {}


def optimizer_register(name):
    def decorator(func_or_class):
        if name in optimizer_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        optimizer_registry[name] = func_or_class
        return func_or_class
    return decorator

def benchmark_register(name):
    def decorator(func_or_class):
        if name in problem_registry:
            # raise ValueError(f"Error: '{name}' is already registered.")
            pass
        problem_registry[name] = func_or_class
        return func_or_class
    return decorator

def normalizer_register(name):
    def decorator(func_or_class):
        if name in normalizer_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        normalizer_registry[name] = func_or_class
        return func_or_class
    return decorator

def acf_register(name):
    def decorator(func_or_class):
        if name in acf_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        acf_registry[name] = func_or_class
        return func_or_class
    return decorator

def para_register(name):
    def decorator(parameter_type):
        if name in para_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        para_registry[name] = parameter_type
        return parameter_type
    return decorator