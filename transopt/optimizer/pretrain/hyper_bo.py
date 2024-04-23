from agent.registry import pretrain_register




@pretrain_register("hyperbo")
class BoxRefine():
    def __init__(self, config) -> None:
        super().__init__(config)