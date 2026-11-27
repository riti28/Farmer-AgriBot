# Farmer AgriBot - Integrated Pest Management (IPM) module

Part of the group project *Farmer AgriBot*. This folder is the **IPM** part. It works for any location in the world:

1. **Pest Risk Predictor**: a small **neural network** predicts this week's pest risk (Low / Medium / High) for
   rice, maize, cotton, sugarcane or finger millet from crop stage and weather. The weather can be typed in or loaded
   live for **any latitude/longitude** from NASA POWER (free, no API key).
2. **Ask IPM**: a **RAG** assistant answers everyday farmer questions ("worms eating my maize") only from 49 IPM
   notes (FAO, ICAR, IRRI, CIMMYT, CABI, ICRISAT, USDA, UC IPM, agricultural universities) and shows citations.
   It uses **hybrid search** (meaning embeddings from `BAAI/bge-small-en-v1.5` via fastembed + TF-IDF words) and a
   farmer-word synonym list. If nothing matches, it says *"Not in my documents"*, suggests 3 close topics and
   points to the local KVK. An LLM can optionally be switched on (see below); it is never required.

> Not an agronomist. No yield guarantee. Consult your local agricultural extension officer (e.g. KVK in India).

No API keys are needed anywhere. The first run downloads the small embedding model (~70 MB, free, no key).

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
| 4 | `python rag.py "your question"` | Asks the RAG assistant from the command line (first run builds `models/chunk_embeddings.npz`) | printed answer |
| 5 | `python eval.py` | OLD vs NEW search on 2 question sets (58 questions) | `eval/results.md` |
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

**Ask IPM: OLD (TF-IDF) vs NEW (hybrid)** (`eval/results.md`, no LLM used):

| Question set | System | Hit@3 | Answered | Off-topic refused | Grounded |
|---|---|---|---|---|---|
| Original questions (27 + 3 off-topic) | OLD: TF-IDF, 29 notes | 1.00 | 1.00 | 3/3 | 0.97 |
| Original questions (27 + 3 off-topic) | **NEW: hybrid + synonyms, 49 notes** | 1.00 | 1.00 | 3/3 | **0.87** |
| Farmer questions (25 + 3 off-topic) | OLD: TF-IDF, 29 notes | 0.32 | 0.40 | 3/3 | 0.29 |
| Farmer questions (25 + 3 off-topic) | TF-IDF + the 20 new notes | 0.92 | 0.68 | 3/3 | 0.68 |
| Farmer questions (25 + 3 off-topic) | **NEW: hybrid + synonyms, 49 notes** | **0.96** | **1.00** | 3/3 | **0.79** |

*Hit@3* = a correct note is in the top 3. *Answered* = the bot did not say "Not in my documents".
*Grounded* = the answer cites a correct note **and** contains the key fact, or an off-topic question is refused.

Honest reading of these numbers:
- On everyday farmer questions the NEW system is much better (Hit@3 0.32 → 0.96). Most of the gain comes from the
  20 new notes; hybrid search and synonyms add the rest (answered 0.68 → 1.00, grounded 0.68 → 0.79).
- On the original questions the grounded score **drops** (0.97 → 0.87): the NEW answer shows only the 2-4 best
  sentences, so it sometimes leaves out the exact key fact that the old whole-chunk answer contained.
- We wrote both question sets, and the new notes target everyday problems. The threshold (0.47) was picked from these
  same questions; the gap between the highest off-topic score (0.454) and the lowest answerable score (0.487) is small,
  so some real off-topic questions will slip through. Known miss: "Trichogramma" retrieves the "Trichoderma" note.

## Important limitations
- Labels are **rule-derived from published weather thresholds, not real field pest counts**.
- The thresholds come mostly from Indian studies; check them before using the tool in other countries.
- One weather point per site; no soil, variety, irrigation or pesticide data. See `data/DATA_DICTIONARY.md`.
- Corpus URLs marked **VERIFY** in `corpus/SOURCES.md` must be checked by hand.

## Optional: switch on an LLM for nicer answers (not required)
Without a key the app shows the 2-4 best sentences from our notes (the default, fully free). With a free
[Groq](https://console.groq.com) key, an LLM rewrites the answer **only from the retrieved passages**, with [1][2][3]
citations. For questions our notes do not cover it may give short general info, labelled
"⚠️ Not from our documents – general info, confirm with your KVK". If the key is missing or wrong, the app quietly
falls back to the free mode.
1. Create a free account at https://console.groq.com and make an API key (API Keys → Create API Key).
2. **Local:** `set GROQ_API_KEY=your_key` (Windows) or `export GROQ_API_KEY=your_key` (Mac/Linux), then `streamlit run app.py`.
3. **Streamlit Cloud:** open the app → **Manage app** → **⋮ → Settings → Secrets**, paste `GROQ_API_KEY = "your_key"`, **Save**.
4. Optional: `GROQ_MODEL = "llama-3.1-8b-instant"` (default) can be changed to any model Groq lists.
Never commit the key to GitHub.

## MCP server (bonus)

`mcp_server.py` exposes two tools over MCP (stdio), built with FastMCP:

- `pest_risk(crop, crop_stage, temp, humidity, rainfall)` → risk, confidence, probabilities
- `search_ipm_knowledge(query)` → 2-4 best sentences from the top-3 passages + citations (+ suggestions if nothing matched)

**Quick test (no client needed):**

```bash
python mcp_server.py --test
```

Expected output (numbers may change slightly if you retrain):

```
{'risk': 'High', 'confidence': 0.81, 'probabilities': {'High': 0.81, 'Low': 0.058, 'Medium': 0.132}, 'disclaimer': '...'}
{'answer': 'Caterpillars ("worms") are the most common pests of maize. [1] Earworms (Helicoverpa) feed on cobs. [2]', 'citations': [{'doc_id': 'ipm47_maize_pests_overview', ...}], 'suggestions': []}
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
**Live app: https://farmer-agribot-ipm.streamlit.app/** (deployed from branch `ipm`, Python 3.13).

Main file path: `ipm/app.py`. Streamlit Cloud uses `ipm/requirements.txt` because it sits next to the main file.
Step-by-step instructions are in the PR description and in `LEARN.md`.

## Folder map
```
ipm/
  fetch_data.py  make_dataset.py  train.py  predict.py  rag.py  eval.py  app.py  mcp_server.py
  data/      weather_raw.csv, pest_data.csv, DATA_DICTIONARY.md
  corpus/    49 IPM notes (*.txt) + SOURCES.md
  models/    best_model.joblib, chunk_embeddings.npz (cached note embeddings)
  results/   results.csv, confusion_*.png
  eval/      questions.csv, farmer_questions.csv, results.md
  requirements.txt  README.md  LEARN.md
```
