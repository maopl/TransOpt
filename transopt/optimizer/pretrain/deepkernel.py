from transopt.agent.registry import pretrain_registry
from transopt.optimizer.pretrain.pretrain_base import PretrainBase



@pretrain_registry.register("deepkernel")
class DeepKernelPretrain(PretrainBase):
    def __init__(self, config) -> None:
        super().__init__(config)