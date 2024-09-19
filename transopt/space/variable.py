import math


class Variable:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    @property
    def search_space_range(self):
        raise NotImplementedError

    def map2(self, value):
        raise NotImplementedError

    def map_inverse(self, value):
        raise NotImplementedError


class Continuous(Variable):
    def __init__(self, name, range):
        super().__init__(name, "continuous")
        self.range = range
        
        self.is_discrete = False

    @property
    def search_space_range(self):
        return self.range

    def map2(self, value):
        return value
    
    def map_inverse(self, value):
        return value


class Categorical(Variable):
    def __init__(self, name, categories):
        super().__init__(name, "categorical")
        self.categories = categories
        self.range = (1, len(self.categories))
        
        self.is_discrete = True

    @property
    def search_space_range(self):
        return (1, len(self.categories))

    def map2(self, value):
        return self.categories.index(value) + 1

    def map_inverse(self, value):
        return self.categories[int(value) - 1]


class Integer(Variable):
    def __init__(self, name, range):
        super().__init__(name, "integer")
        self.range = range

        self.is_discrete = True

    @property
    def search_space_range(self):
        return self.range

    def map2(self, value):
        return value

    def map_inverse(self, value):
        return round(value)


class LogContinuous(Variable):
    def __init__(self, name, range):
        super().__init__(name, "log_continuous")
        self.range = range
        
        self.is_discrete = False

    @property
    def search_space_range(self):
        return self.range

    def map2(self, value):
        return 10**value

    def map_inverse(self, value):
        return math.log10(value)

