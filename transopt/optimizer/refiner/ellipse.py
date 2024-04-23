from transopt.optimizer.refiner.refiner_base import RefinerBase
from transopt.agent.registry import space_refine_registry

@space_refine_registry.register("ellipse")
class EllipseRefiner(RefinerBase):
    def __init__(self, config) -> None:
        super().__init__(config)