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
conda run -n visual-agent python pipelines/run.py pipelines/configs/<name>.yaml
```

Output is written to `output/<pipeline-name>/`. If the last stage is `EvalResult`, accuracy is printed at the end.

---

## Configs

All configs live in `pipelines/configs/`. Each YAML defines a loader, stages, and processors.

---

## Viewer

Browse pipeline outputs in a Streamlit app:

```bash
conda run -n visual-agent streamlit run pipelines/viewer.py
```

---

## Data

Images live in `data/` and can be downloaded using `scripts/0330_download_images.py`. 

The canonical dataset is `labels/cb_412.json` (412 items), built from CountBench via:

```bash
conda run -n visual-agent python scripts/0418_prepare_countbench.py
```
