"""Count objects from ObjectCoordinates by counting the number of points."""
from agentflow.processors.base import Processor
from ..types import ObjectCoordinates, VLMCount


class CoordinateCountProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> VLMCount:
        coords: ObjectCoordinates = inputs[self._input_names_snake[0]]
        count = len(coords.points)
        if logger:
            print(f"num_points: {count}", file=logger, flush=True)
        return VLMCount(count=count)
