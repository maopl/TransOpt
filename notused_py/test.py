import ConfigSpace as CS

CS.UniformFloatHyperparameter('eta', lower=0, upper=1., default_value=0)
obj = CS.UniformFloatHyperparameter(name='example', lower=0, upper=10)
print(obj.lower)  # This should output 0
print(obj.upper)  # This should output 10