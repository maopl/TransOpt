class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name=None, cls=None, **kwargs):
        if cls is None:
            def wrapper(cls):
                return self.register(name, cls, **kwargs)
            return wrapper
        
        if name is None:
            name = cls.__name__

        if name in self._registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        
        self._registry[name] = {'cls': cls, **kwargs}
        return cls

    def get(self, name):
        return self._registry[name]['cls']

    def list_names(self):
        return list(self._registry.keys())

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        return item in self._registry

space_refine_register = Registry()
sampler_register = Registry()
pretrain_register = Registry()
model_register = Registry()
acf_register = Registry()
problem_register = Registry()
statistic_register = Registry()

# 示例使用
# print(sampler_registry.list_names())
# print(optimizer_registry["GeneticAlgorithm"].__doc__)