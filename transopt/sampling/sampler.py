
class Sampler:
    """采样器接口，所有采样器都应该继承这个类并实现sample方法。"""
    def sample(self, search_space, n_samples=1):
        raise NotImplementedError("Sample method should be implemented by subclasses.")