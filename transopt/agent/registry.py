# regirstry for optimizer, 
g_space_define_registry = {}
g_initializer_registry = {}
g_pretrain_registry = {}
g_model_registry = {}
g_acf_registry = {}


#registry for benchmark
g_problem_registry = {}
g_statistic_registry = {}




def space_define_register(name):
    def decorator(func_or_class):
        if name in g_space_define_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        g_space_define_registry[name] = func_or_class
        return func_or_class
    return decorator



def initializer_register(name):
    def decorator(func_or_class):
        if name in g_initializer_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        g_initializer_registry[name] = func_or_class
        return func_or_class
    return decorator

def pretrain_register(name):
    def decorator(func_or_class):
        if name in g_pretrain_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        g_pretrain_registry[name] = func_or_class
        return func_or_class
    return decorator

def model_register(name):
    def decorator(func_or_class):
        if name in g_model_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        g_model_registry[name] = func_or_class
        return func_or_class
    return decorator

def acf_register(name):
    def decorator(func_or_class):
        if name in g_acf_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        g_acf_registry[name] = func_or_class
        return func_or_class
    return decorator

def problem_register(name):
    def decorator(func_or_class):
        if name in g_problem_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        g_problem_registry[name] = func_or_class
        return func_or_class
    return decorator


def statistic_register(name):
    def decorator(func_or_class):
        if name in g_statistic_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        g_statistic_registry[name] = func_or_class
        return func_or_class
    return decorator

