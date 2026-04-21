"""Per-pipeline runner invoked by run_all.py as a subprocess.

Usage:
    python pipelines/_runner.py <pipeline_yaml_name>

Runs from visual-agent/ as cwd.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VISUAL_AGENT = _HERE.parent

for _p in [str(_VISUAL_AGENT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml
from agentflow.pipeline import Pipeline
from agentflow.typing.config import Config

# --- Register types ---
from pipelines.types import DinoPrompt, DetectionResult, SAMMasks, ObjectCoordinates, EvalResult, VLMCount

Pipeline.register_type("DinoPrompt", DinoPrompt)
Pipeline.register_type("DetectionResult", DetectionResult)
Pipeline.register_type("SAMMasks", SAMMasks)
Pipeline.register_type("ObjectCoordinates", ObjectCoordinates)
Pipeline.register_type("EvalResult", EvalResult)
Pipeline.register_type("VLMCount", VLMCount)

# --- Register processors ---
from pipelines.processors.dino_prompt import DinoPromptProcessor
from pipelines.processors.oracle_prompt import OraclePromptProcessor
from pipelines.processors.grounding_dino import GroundingDinoProcessor
from pipelines.processors.sam_box import SAMBoxProcessor
from pipelines.processors.sam_auto import SAMAutoProcessor
from pipelines.processors.clip_verify import CLIPVerifyProcessor
from pipelines.processors.vlm_count import VLMCountProcessor
from pipelines.processors.vlm_locate import VLMLocateProcessor
from pipelines.processors.sam_point import SAMPointProcessor
from pipelines.processors.evaluator import EvaluatorProcessor
from pipelines.processors.mobilenet_detect import MobileNetDetectProcessor
from pipelines.processors.max_class_count import MaxClassCountProcessor

Pipeline.register_processor("DinoPromptProcessor", DinoPromptProcessor)
Pipeline.register_processor("OraclePromptProcessor", OraclePromptProcessor)
Pipeline.register_processor("GroundingDinoProcessor", GroundingDinoProcessor)
Pipeline.register_processor("SAMBoxProcessor", SAMBoxProcessor)
Pipeline.register_processor("SAMAutoProcessor", SAMAutoProcessor)
Pipeline.register_processor("CLIPVerifyProcessor", CLIPVerifyProcessor)
Pipeline.register_processor("VLMCountProcessor", VLMCountProcessor)
Pipeline.register_processor("VLMLocateProcessor", VLMLocateProcessor)
Pipeline.register_processor("SAMPointProcessor", SAMPointProcessor)
Pipeline.register_processor("EvaluatorProcessor", EvaluatorProcessor)
Pipeline.register_processor("MobileNetDetectProcessor", MobileNetDetectProcessor)
Pipeline.register_processor("MaxClassCountProcessor", MaxClassCountProcessor)

# --- Load config ---
yaml_name = sys.argv[1]
config_path = Path(yaml_name) if Path(yaml_name).is_absolute() else Path.cwd() / yaml_name
with open(config_path) as f:
    raw = yaml.safe_load(f)

if raw['name'] != config_path.stem:
    input(f"{raw['name']} vs {config_path.stem}. continue?" )

raw["loader"]["source"] = str(_VISUAL_AGENT / raw["loader"]["source"])
raw["loader"]["kwargs"]["image_dir"] = str(_VISUAL_AGENT / raw["loader"]["kwargs"]["image_dir"])

cfg = Config.model_validate(raw)
prompt_dir = str(_HERE / "prompts")
pipeline = Pipeline(cfg, prompt_dir=prompt_dir)

print(f"Running pipeline: {cfg.name} ({len(pipeline.item_ids)} items)")
pipeline.execute_all()
print(f"Done: {cfg.name}")

if cfg.stages[-1].output == "EvalResult":
    from pathlib import Path as _Path
    import json as _json
    eval_dir = _Path("output") / cfg.name / "EvalResult"
    results = [_json.loads(p.read_text()) for p in sorted(eval_dir.glob("*/output.json"))]
    if results:
        correct = sum(r["is_correct"] for r in results)
        total = len(results)
        print(f"\n=== Evaluation: {correct}/{total} = {correct/total:.1%} ===")
