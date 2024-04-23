from transopt.agent.registry import g_pretrain_registry



def get_pretrain(pretrain_name, **kwargs):
    """Create the optimizer object."""
    pretrain_class = g_pretrain_registry.get(pretrain_name)
    config = kwargs

    if pretrain_class is not None:
        pretrain_method = pretrain_class(config=config)
    else:
        print(f"Refiner '{pretrain_name}' not found in the registry.")
        raise NameError
    return pretrain_method