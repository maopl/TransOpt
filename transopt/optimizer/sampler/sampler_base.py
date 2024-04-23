
class Sampler:
    def __init__(self, config) -> None:
        self.config = config
        
    def sample(self, search_space, n_samples=1, metadata=None):
        raise NotImplementedError("Sample method should be implemented by subclasses.")
    
    def check_metadata_avaliable(self, metadata):
        if metadata is None:
            return False
        return True