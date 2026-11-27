# Farmer AgriBot - Integrated Pest Management (IPM) module

Part of the group project *Farmer AgriBot*. This folder is the **IPM** part. It works for any location in the world:

1. **Pest Risk Predictor**: a small **neural network** predicts this week's pest risk (Low / Medium / High) for
   rice, maize, cotton, sugarcane or finger millet from crop stage and weather. The weather can be typed in or loaded
   live for **any latitude/longitude** from NASA POWER (free, no API key).
2. **Ask IPM**: a **RAG** assistant (TF-IDF retrieval) answers only from 29 IPM notes (FAO, ICAR, IRRI, CIMMYT, CABI,
   ICRISAT, USDA, agricultural universities) and shows citations. If nothing matches, it says *"Not in my documents."*

> Not an agronomist. No yield guarantee. Consult your local agricultural extension officer (e.g. KVK in India).

No API keys are needed anywhere.

## Setup

```bash
cd ipm
python -m venv .venv
# Windows: .venv\Scripts\activate     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 on Windows.

## Run order

The data, model and results are already committed, so you can jump straight to step 6.

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python fetch_data.py` | Downloads 2015-2024 daily weather for 16 sites on 5 continents from NASA POWER, makes it weekly | `data/weather_raw.csv` |
| 2 | `python make_dataset.py` | Labels each crop-week Low/Medium/High with published thresholds, adds 10% noise | `data/pest_data.csv` |
| 3 | `python train.py` | Trains a dummy baseline and 2 neural networks, tests on unseen sites, saves the best | `results/`, `models/best_model.joblib` |
| 4 | `python rag.py "your question"` | Asks the RAG assistant from the command line | printed answer |
| 5 | `python eval.py` | Hit@3 and grounded-answer score on 30 questions | `eval/results.md` |
| 6 | `streamlit run app.py` | Opens the web app | browser |
| 7 | `python mcp_server.py --test` | Calls both MCP tools once | printed result |

## Results

**Pest risk model** (stratified 80/20 split, 3,740 rows; `results/results.csv`):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Dummy baseline (always "Low") | 0.531 | 0.231 |
| Neural net (1 hidden layer, 32 neurons) | 0.757 | 0.711 |
| **Deep neural net (64-32-16 neurons)** | **0.805** | **0.771** |
| Deep neural net, tested on 3 never-seen sites (Can Tho, Lubbock, Soroti) | 0.686 | 0.669 |

With 10% random label noise, about 0.90 accuracy is the best any model could reach.
Confusion matrices: `results/confusion_*.png`.

**RAG assistant** (`eval/results.md`):

| Metric | Score |
|---|---|
| Hit@3 (27 answerable questions) | 1.00 |
| Grounded-answer score (30 questions, incl. 3 off-topic) | 0.97 |

These RAG scores are optimistic: we wrote the questions from our own notes and chose the threshold (0.15) by looking at them.

## Important limitations
- Labels are **rule-derived from published weather thresholds, not real field pest counts**.
- The thresholds come mostly from Indian studies; check them before using the tool in other countries.
- One weather point per site; no soil, variety, irrigation or pesticide data. See `data/DATA_DICTIONARY.md`.
- Corpus URLs marked **VERIFY** in `corpus/SOURCES.md` must be checked by hand.

## MCP server (bonus)

`mcp_server.py` exposes two tools over MCP (stdio), built with FastMCP:

- `pest_risk(crop, crop_stage, temp, humidity, rainfall)` → risk, confidence, probabilities
- `search_ipm_knowledge(query)` → answer from the top-3 passages + citations

**Quick test (no client needed):**

```bash
python mcp_server.py --test
```

Expected output (numbers may change slightly if you retrain):

```
{'risk': 'High', 'confidence': 0.81, 'probabilities': {'High': 0.81, 'Low': 0.058, 'Medium': 0.132}, 'disclaimer': '...'}
{'answer': 'Some synthetic pyrethroids can cause BPH resurgence ... [1] ...', 'citations': [{'doc_id': 'ipm04_rice_bph', ...}]}
```

**Test with the MCP Inspector:** `mcp dev mcp_server.py`, open the link it prints, click *Tools → pest_risk* and
enter `crop=cotton, crop_stage=flowering, temp=22, humidity=55, rainfall=0`.

**Use from Claude Desktop:** add this to `claude_desktop_config.json` (use your own full paths):

```json
{
  "mcpServers": {
    "agribot-ipm": {
      "command": "C:/full/path/to/python.exe",
      "args": ["C:/full/path/to/Farmer-AgriBot/ipm/mcp_server.py"]
    }
  }
}
```

## Deploy on Streamlit Community Cloud
Main file path: `ipm/app.py`. Streamlit Cloud uses `ipm/requirements.txt` because it sits next to the main file.
Step-by-step instructions are in the PR description and in `LEARN.md`.

## Folder map
```
ipm/
  fetch_data.py  make_dataset.py  train.py  predict.py  rag.py  eval.py  app.py  mcp_server.py
  data/      weather_raw.csv, pest_data.csv, DATA_DICTIONARY.md
  corpus/    29 IPM notes (*.txt) + SOURCES.md
  models/    best_model.joblib
  results/   results.csv, confusion_*.png
  eval/      questions.csv, results.md
  requirements.txt  README.md  LEARN.md
```
