from transopt.agent.registry import g_pretrain_registry



def get_pretrain(pretrain, **kwargs):
    """Create the optimizer object."""
    pretrain_class = g_pretrain_registry.get(pretrain)
    config = kwargs

    if pretrain_class is not None:
        pretrain = pretrain_class(config=config)
    else:
        print(f"Refiner '{pretrain}' not found in the registry.")
        raise NameError
    return pretrain