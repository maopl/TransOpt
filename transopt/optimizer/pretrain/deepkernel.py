from transopt.agent.registry import pretrain_register
from transopt.optimizer.pretrain.pretrain_base import PretrainBase



@pretrain_register.register("deepkernel")
class DeepKernelPretrain(PretrainBase):
    def __init__(self, config) -> None:
        super().__init__(config)