# Visual Agent

Agentic computer vision system for **object counting** and **object search**, using:

- **Grounding DINO** — zero-shot object detection
- **CLIP ViT-L/14** — semantic verification and image similarity
- **LangChain** — tool-calling agent loop (ReAct pattern)
- **MiniMax M2.7** — LLM orchestrator (via MiniMax API)
- **Streamlit** — chat UI with live tool call visibility

---

## Setup

### 1. Clone and create the conda environment

```bash
git clone <repo-url>
cd visual-agent
conda create -n visual-agent python=3.11 -y
conda activate visual-agent
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
```

Edit `.env` and fill in your MiniMax API key:

```
MINIMAX_API_KEY=your-minimax-api-key
```

`MODEL_ID` defaults to `MiniMax-M2.7`. Override it by setting `MODEL_ID` in `.env`.

### 3. Run

```bash
conda run -n visual-agent streamlit run app.py
```

Models are downloaded from HuggingFace on first run (~1.3 GB total, cached automatically).

---

## Usage

### Object Counting

1. Select **Object Counting** in the sidebar
2. Upload a scene image
3. Type a question: `How many people are in this image?`

The agent will:
1. Detect candidates with Grounding DINO
2. Verify detections with CLIP
3. Annotate verified objects with lime boxes
4. Report the count

### Object Search

1. Select **Object Search** in the sidebar
2. Upload a scene image and a reference/pattern image
3. Type a question: `Find this person in the crowd`

The agent will:
1. Detect all relevant objects in the scene
2. Rank them by CLIP similarity to the reference image
3. Show a grid of the top matches with similarity scores

---

## Project Structure

```
visual-agent/
├── app.py                        # Streamlit UI
├── agent/
│   ├── agent.py                  # LangChain agent factory
│   └── tools/
│       ├── model_registry.py     # Singleton lazy loader for GDINO + CLIP
│       ├── detection_tools.py    # Grounding DINO tool
│       ├── similarity_tools.py   # CLIP verify + rank tools
│       └── image_tools.py        # Grid and annotation tools
├── test_count_people.py          # Unit test: vision pipeline only
├── test_agent_count_people.py    # End-to-end test: full agent + LLM
├── requirements.txt
└── .env.example
```

---

## Tests

```bash
# Unit test — runs Grounding DINO + CLIP directly, no LLM needed
conda run -n visual-agent python test_count_people.py

# End-to-end test — runs the full agent (requires MINIMAX_API_KEY)
conda run -n visual-agent python test_agent_count_people.py
```

---

## Notes

- All temporary files (crops, annotated images) are saved to `./tmp/` and gitignored
- Vision models run on GPU if available, otherwise CPU
- CLIP similarity thresholds: `0.15` for people, `0.25` for other objects
