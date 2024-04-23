from agent.registry import g_sampler_registry



def get_sampler(sampler_name, **kwargs):
    """Create the optimizer object."""
    sampler_class = g_sampler_registry.get(sampler_name)
    config = kwargs

    if sampler_class is not None:
        sampler = sampler_class(config=config)
    else:
        print(f"Sampler '{sampler_name}' not found in the registry.")
        raise NameError
    return sampler