from refiner.refiner_base import RefinerBase
from agent.registry import space_refine_register

@space_refine_register("ellipse")
class EllipseRefine(RefinerBase):
    def __init__(self, config) -> None:
        super().__init__(config)