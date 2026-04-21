# Visual Agent

Pipeline framework for **object counting** benchmarks, built on [agentflow](https://github.com/npnkhoi/agentflow).

---

## Setup

```bash
git clone git@github.com:npnkhoi/visual-agent.git
cd visual-agent
conda create -n visual-agent python=3.11 -y
conda activate visual-agent
pip install -r requirements.txt
pip install git+https://github.com/npnkhoi/agentflow.git
```

---

## Running a pipeline

```bash
conda run -n visual-agent python pipelines/_runner.py pipelines/configs/<name>.yaml
```

Output is written to `output/<pipeline-name>/`. If the last stage is `EvalResult`, accuracy is printed at the end.

---

## Configs

All configs live in `pipelines/configs/`. Each YAML defines a loader, stages, and processors.

| Config | Description |
|---|---|
| `gdino_tiny.yaml` | Grounding DINO tiny, oracle noun prompt |
| `gdino_tiny_prompt_extract.yaml` | Grounding DINO tiny, LLM noun extraction |
| `gdino_base.yaml` | Grounding DINO base, oracle noun prompt |
| `gdino_base_prompt_extract.yaml` | Grounding DINO base, LLM noun extraction |
| `mobilenet.yaml` | SSDLite MobileNetV3, max-class heuristic |
| `vlm_count_qwen.yaml` | Qwen2.5-VL-7B direct counting |
| `vlm_count_gemma.yaml` | Gemma-4 direct counting |

---

## Viewer

Browse pipeline outputs in a Streamlit app:

```bash
conda run -n visual-agent streamlit run pipelines/viewer.py
```

---

## Data

Images live in `data/`. The canonical dataset is `pipelines/data/countbench.json` (412 items), built from CountBench via:

```bash
conda run -n visual-agent python scripts/0418_prepare_countbench.py
```
