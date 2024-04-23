from agent.registry import pretrain_register




@pretrain_register("deepkernel")
class BoxRefine():
    def __init__(self, config) -> None:
        super().__init__(config)