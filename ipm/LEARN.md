# LEARN.md: how this IPM module works, in simple words

## The big picture

```
NASA POWER weather ──> fetch_data.py ──> weather_raw.csv
                                              │
        published pest thresholds ──> make_dataset.py ──> pest_data.csv (labelled)
                                                                │
                                                           train.py ──> best_model.joblib (neural network)
                                                                │
   user picks crop + stage + weather (or a location) ──> predict.py ──> risk + confidence
                                                                │
 29 IPM notes (corpus/) ──> rag.py (TF-IDF search) ──> tips & answers with citations
                                                                │
                                      app.py (Streamlit)  and  mcp_server.py (MCP tools)
```

1. **Data**: we download real weather and turn it into weekly numbers.
2. **Labels**: scientists have published which weather helps each pest. We turn those findings into rules and use the
   rules to label every crop-week Low / Medium / High. We add 10% random mistakes so the data is not "too perfect".
3. **Train**: a neural network learns to copy those labels from crop, stage, temperature, humidity and rain.
4. **Predict**: for any new week of weather, anywhere in the world, the network gives a risk and how sure it is.
5. **RAG**: for advice, we do not let the computer invent text. It *retrieves* the most similar passages from our own
   notes and shows them with the source. If nothing is similar enough, it says "Not in my documents."

## Every file explained

| File | What it does |
|---|---|
| `fetch_data.py` | Calls the free NASA POWER API for 16 towns (India, Philippines, Vietnam, Zambia, Brazil, USA, Pakistan, Australia, Uganda, Nepal), 2015-2024. Averages daily temperature and humidity per week and adds up the rain. Also has `fetch_daily()`, reused by the app to get live weather for any latitude/longitude. |
| `make_dataset.py` | Holds `THRESHOLDS` (one readable dict, one entry per pest, each with its paper citation) and `CROP_CALENDAR` (which site grows which crop and the sowing week). For every crop-week it checks the rules: all met = High, one missed = Medium, more missed = Low. Adds 10% label noise. |
| `train.py` | Splits data 80/20 keeping the same Low/Medium/High mix (stratified). Turns text columns into 0/1 columns (one-hot) and scales numbers. Trains a dummy baseline, a 1-layer neural net and a 3-layer "deep" neural net (scikit-learn `MLPClassifier`). Prints accuracy and macro-F1, saves confusion matrix pictures, re-tests the best model on 3 sites it never saw, and saves the best model. |
| `predict.py` | Loads the saved model. `predict_risk()` gives risk + confidence. `latest_week_weather()` gets the last 7 days of real weather for any point on Earth. `ipm_tips()` searches the notes for the crop's pests. |
| `rag.py` | Reads the notes, cuts each into 3-sentence chunks, builds TF-IDF vectors, and finds the 3 chunks most similar to a question (cosine similarity). Answers only with those chunks plus [1] [2] [3] citations. Score below 0.15 → "Not in my documents." |
| `eval.py` | Runs 30 test questions. **Hit@3** = was a correct note in the top 3? **Grounded score** = did the answer cite a correct note *and* contain the key fact (or correctly refuse off-topic questions)? Writes `eval/results.md`. |
| `app.py` | Streamlit web page with two tabs: *Pest Risk Predictor* and *Ask IPM* (chat), plus the fixed disclaimer. |
| `mcp_server.py` | Makes two tools (`pest_risk`, `search_ipm_knowledge`) available to AI assistants through the Model Context Protocol, using FastMCP. |
| `data/DATA_DICTIONARY.md` | Every column, unit and source, the threshold table with citations, and the limitations. |
| `corpus/*.txt`, `corpus/SOURCES.md` | 29 short IPM notes in our own words with title, URL and organisation; the list of sources and which URLs still need checking. |
| `eval/questions.csv` | 30 questions, the note(s) that should answer each, and a key word the answer should contain. |
| `models/best_model.joblib` | The saved neural network (with its preprocessing), so the app does not need to retrain. |
| `results/` | `results.csv` scores and confusion matrix pictures. |
| `requirements.txt` | Exact library versions, so the code runs the same on every computer. |

## Key ideas in one line each
- **Neural network (MLP)**: layers of small "neurons"; each multiplies inputs by weights, adds them, and passes the
  result through an activation function. Training adjusts the weights to reduce mistakes (backpropagation).
- **Deep** = more than one hidden layer (ours: 64 → 32 → 16).
- **One-hot encoding**: "rice" becomes a column rice=1, maize=0, cotton=0, ...
- **Scaling**: puts temperature (≈25) and rain (≈0-300) on similar scales so no input dominates.
- **Stratified split**: the test set has the same share of Low/Medium/High as the training set.
- **Macro-F1**: average of the F1 score of each class, so the rare "High" class counts as much as "Low".
- **Confusion matrix**: table of true label vs predicted label; the diagonal shows correct answers.
- **TF-IDF**: a word gets a high weight if it is frequent in this chunk but rare in other chunks.
- **Cosine similarity**: the angle between two vectors; 1 = same direction (very similar), 0 = nothing in common.
- **RAG**: Retrieval-Augmented Generation. Retrieve first, then answer only from what was retrieved, which reduces made-up answers.

## Deploying on Streamlit Community Cloud (click-by-click)
1. Make sure the `ipm` branch is merged into `main` on GitHub (or deploy straight from the `ipm` branch).
2. Go to https://share.streamlit.io and click **Continue with GitHub**. Sign in to GitHub and click **Authorize streamlit**.
3. Click **Create app** (top right) → choose **Deploy a public app from GitHub** (or "Yup, I have an app").
4. Repository: `riti28/Farmer-AgriBot`. Branch: `main` (or `ipm`). Main file path: `ipm/app.py`.
5. Optional: App URL, e.g. `farmer-agribot-ipm`.
6. Click **Advanced settings** → Python version **3.13** (3.12 also works) → **Save**.
7. Click **Deploy**. The first build takes a few minutes. Streamlit Cloud installs `ipm/requirements.txt` automatically.
8. If you are not an owner of `riti28/Farmer-AgriBot`, ask the owner to deploy, or fork the repo and deploy your fork.

## 15 likely viva questions (with answers)

1. **What is IPM?** Keeping pests below the economic damage level by combining cultural, mechanical, biological and,
   only if needed, chemical methods, instead of routine spraying.
2. **Where does your weather data come from?** NASA POWER daily API: T2M (temperature), RH2M (humidity) and
   PRECTOTCORR (rain), 2015-2024, for 16 sites on 5 continents. It is free, global and needs no key.
3. **Are your labels real pest observations?** No. They are rule-derived from published weather-pest thresholds.
   The model learns those rules. This is the biggest limitation and we state it clearly.
4. **Why add 10% label noise?** Real data is never perfect. Noise stops the model from scoring an unrealistic 100%
   and shows how it behaves with mistakes. It also caps the best possible accuracy at about 90%.
5. **Why a neural network and not a decision tree?** The project focus is AI/deep learning. A neural network learns
   non-linear combinations of weather and crop stage. We compare it against a dummy baseline to prove it learned something.
6. **What is the dummy baseline and why include it?** It always predicts the most common class ("Low"). Accuracy 0.53
   but macro-F1 only 0.23. Any real model must beat it.
7. **Why macro-F1 and not only accuracy?** Classes are imbalanced (Low is most common). Accuracy can look good by
   always saying "Low"; macro-F1 treats High, Medium and Low equally.
8. **How do you know it works outside the training places?** We trained without Can Tho, Lubbock and Soroti and
   tested only on them: macro-F1 0.67 (vs 0.77 on the normal test). It drops but stays far above the baseline.
9. **Why are site and week not model inputs?** So the model is location-free and can be used anywhere; seasons differ
   by hemisphere, so week number would mislead.
10. **What is RAG and why use it?** Retrieval-Augmented Generation: find relevant passages first, then answer only from
    them. It keeps answers grounded in trusted sources and lets us show citations.
11. **How does TF-IDF + cosine similarity work?** Each chunk and the question become word-weight vectors; the chunks
    whose vectors point most in the same direction as the question are the top results.
12. **What happens with an off-topic question?** If the best similarity is below 0.15 the bot replies "Not in my
    documents." All 3 off-topic test questions were refused correctly.
13. **What are Hit@3 and grounded-answer score?** Hit@3: share of answerable questions where a correct note is in the
    top 3 (1.00). Grounded: the answer cites a correct note and contains the key fact, or correctly refuses (0.97).
14. **What are the limitations?** Rule-based labels, thresholds mostly from Indian studies, one weather point per site,
    fixed sowing weeks, no soil/variety/pesticide data, small self-written corpus and evaluation questions.
15. **What is the MCP server for?** It lets AI assistants such as Claude Desktop call our two tools (pest risk and IPM
    search) in a standard way, so the module can plug into a larger chatbot.
