"""Pick the most-detected class from a DetectionResult and return its count."""
from collections import Counter

from agentflow.processors.base import Processor
from ..types import DetectionResult, VLMCount


class MaxClassCountProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> VLMCount:
        detection: DetectionResult = inputs[self._input_names_snake[0]]

        if not detection.labels:
            count = 0
            top_label = None
        else:
            label_counts = Counter(detection.labels)
            top_label, count = label_counts.most_common(1)[0]

        if logger:
            print(f"top_label: {top_label}  count: {count}", file=logger, flush=True)

        return VLMCount(count=count)
