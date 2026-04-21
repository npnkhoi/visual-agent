"""Stage: Compare predicted count against the ground-truth answer."""
from agentflow.processors.base import Processor
from ..types import EvalResult


class EvaluatorProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> EvalResult:
        ground_truth = int(inputs[self._input_names_snake[0]])
        result = inputs[self._input_names_snake[1]]
        if hasattr(result, "num_boxes"):
            predicted = result.num_boxes
        elif hasattr(result, "count"):
            predicted = result.count
        else:
            predicted = int(result)
        is_correct = predicted == ground_truth
        if logger:
            print(
                f"predicted={predicted}  ground_truth={ground_truth}  correct={is_correct}",
                file=logger,
                flush=True,
            )
        return EvalResult(
            predicted=predicted,
            ground_truth=ground_truth,
            is_correct=is_correct,
        )
