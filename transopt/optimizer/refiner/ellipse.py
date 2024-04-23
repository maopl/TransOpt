from transopt.optimizer.refiner.refiner_base import RefinerBase
from transopt.agent.registry import space_refine_register

@space_refine_register.register("ellipse")
class EllipseRefiner(RefinerBase):
    def __init__(self, config) -> None:
        super().__init__(config)