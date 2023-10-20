
# 定义一个空的注册表字典
optimizer_registry = {}
benchmark_registry = {}
accessor_registry = {}
acf_registry = {}


# 注册函数的装饰器
def optimizer_register(name):
    def decorator(func_or_class):
        if name in optimizer_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        optimizer_registry[name] = func_or_class
        return func_or_class
    return decorator

# 注册函数的装饰器
def benchmark_register(name):
    def decorator(func_or_class):
        if name in benchmark_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        benchmark_registry[name] = func_or_class
        return func_or_class
    return decorator

def accessor_register(name):
    def decorator(func_or_class):
        if name in accessor_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        accessor_registry[name] = func_or_class
        return func_or_class
    return decorator

def acf_register(name):
    def decorator(func_or_class):
        if name in accessor_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        accessor_registry[name] = func_or_class
        return func_or_class
    return decorator