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
 49 IPM notes (corpus/) ──> rag.py (synonyms + hybrid search) ──> best sentences with citations
                                                                │
                                      app.py (Streamlit)  and  mcp_server.py (MCP tools)
```

1. **Data**: we download real weather and turn it into weekly numbers.
2. **Labels**: scientists have published which weather helps each pest. We turn those findings into rules and use the
   rules to label every crop-week Low / Medium / High. We add 10% random mistakes so the data is not "too perfect".
3. **Train**: a neural network learns to copy those labels from crop, stage, temperature, humidity and rain.
4. **Predict**: for any new week of weather, anywhere in the world, the network gives a risk and how sure it is.
5. **RAG**: for advice, we do not let the computer invent text. It *retrieves* the most similar passages from our own
   notes and shows the 2-4 best sentences with the source. If nothing is similar enough, it says "Not in my documents",
   suggests 3 close topics and points to the local KVK. (An LLM can be switched on with a free key; never required.)

## Every file explained

| File | What it does |
|---|---|
| `fetch_data.py` | Calls the free NASA POWER API for 16 towns (India, Philippines, Vietnam, Zambia, Brazil, USA, Pakistan, Australia, Uganda, Nepal), 2015-2024. Averages daily temperature and humidity per week and adds up the rain. Also has `fetch_daily()`, reused by the app to get live weather for any latitude/longitude. |
| `make_dataset.py` | Holds `THRESHOLDS` (one readable dict, one entry per pest, each with its paper citation) and `CROP_CALENDAR` (which site grows which crop and the sowing week). For every crop-week it checks the rules: all met = High, one missed = Medium, more missed = Low. Adds 10% label noise. |
| `train.py` | Splits data 80/20 keeping the same Low/Medium/High mix (stratified). Turns text columns into 0/1 columns (one-hot) and scales numbers. Trains a dummy baseline, a 1-layer neural net and a 3-layer "deep" neural net (scikit-learn `MLPClassifier`). Prints accuracy and macro-F1, saves confusion matrix pictures, re-tests the best model on 3 sites it never saw, and saves the best model. |
| `predict.py` | Loads the saved model. `predict_risk()` gives risk + confidence. `latest_week_weather()` gets the last 7 days of real weather for any point on Earth. `ipm_tips()` searches the notes for the crop's pests. |
| `rag.py` | Reads the notes and cuts each into 3-sentence chunks. Adds synonyms to the question (`SYNONYMS`: "worms" → "larvae caterpillar"). Scores every chunk with **hybrid search** = 0.7 × embedding similarity (meaning) + 0.3 × TF-IDF similarity (words). Chunk embeddings are saved in `models/chunk_embeddings.npz` and rebuilt only when the notes change. From the top-3 chunks it keeps the 2-4 sentences closest to the question, tagged [1] [2] [3]. Best score below 0.47 → "Not in my documents" + 3 suggested topics. If a `GROQ_API_KEY` exists, `ask_llm()` lets an LLM write the answer only from the passages; otherwise (or if it fails) the free sentence answer is used. |
| `eval.py` | Runs 2 question sets (30 note-style + 28 everyday farmer questions) through the OLD TF-IDF search and the NEW hybrid search. **Hit@3** = was a correct note in the top 3? **Answered** = did it avoid "Not in my documents"? **Grounded** = did the answer cite a correct note *and* contain the key fact (or correctly refuse off-topic questions)? Also prints a threshold table. Writes `eval/results.md`. |
| `app.py` | Streamlit web page with two tabs: *Pest Risk Predictor* and *Ask IPM* (chat), plus the fixed disclaimer. |
| `mcp_server.py` | Makes two tools (`pest_risk`, `search_ipm_knowledge`) available to AI assistants through the Model Context Protocol, using FastMCP. |
| `data/DATA_DICTIONARY.md` | Every column, unit and source, the threshold table with citations, and the limitations. |
| `corpus/*.txt`, `corpus/SOURCES.md` | 49 short IPM notes (20 added for everyday problems: aphids, whitefly, thrips, mites, jassids, termites, white grub, rats, when to spray, yellow leaves, neem, Trichoderma, Bt, seed treatment, pesticide labels, dying crops, crop pest overviews) in our own words with title, URL and organisation; the list of sources and which URLs still need checking. |
| `eval/questions.csv`, `eval/farmer_questions.csv` | Test questions, the note(s) that should answer each, and key word(s) the answer should contain ("|" = any of them). |
| `models/chunk_embeddings.npz` | Saved embeddings of every chunk + a fingerprint of the notes, so the app does not re-embed all notes on every start. |
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
- **Embedding**: a list of 384 numbers that a small neural network (`bge-small-en-v1.5`) gives a sentence, so that
  sentences with similar *meaning* get similar numbers, even with different words ("worms" ≈ "caterpillars").
- **fastembed / ONNX**: a light library that runs the embedding model with ONNX Runtime, without PyTorch, so it fits
  on Streamlit's free tier and needs no API key.
- **Hybrid search**: final score = 0.7 × meaning score + 0.3 × word score. Embeddings catch paraphrases; TF-IDF
  catches exact names (e.g. "Trichogramma", "GPU 28") that embeddings can blur.
- **Synonym expansion**: before searching we add note-words for farmer words ("spray" → "insecticide pesticide").
- **Extractive answer**: we copy the best sentences instead of generating new text, so nothing is made up.
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
12. **What happens with an off-topic question?** If the best hybrid score is below 0.47 the bot replies "Not in my
    documents", shows the 3 closest topics we cover as buttons, and says "Ask your local KVK". All 6 off-topic test
    questions were refused correctly.
13. **What are Hit@3 and grounded-answer score?** Hit@3: share of answerable questions where a correct note is in the
    top 3. Grounded: the answer cites a correct note and contains the key fact, or correctly refuses. On everyday farmer
    questions they went from 0.32 / 0.29 (old TF-IDF) to 0.96 / 0.79 (new hybrid).
14. **What are the limitations?** Rule-based labels, thresholds mostly from Indian studies, one weather point per site,
    fixed sowing weeks, no soil/variety/pesticide data, small self-written corpus and evaluation questions.
15. **What is the MCP server for?** It lets AI assistants such as Claude Desktop call our two tools (pest risk and IPM
    search) in a standard way, so the module can plug into a larger chatbot.

### New viva questions about the improved "Ask IPM"
16. **Why was TF-IDF alone not enough?** It only matches the same words. A farmer writes "worms eating my maize" but
    our notes say "caterpillar" and "larvae", so TF-IDF found a correct note in the top 3 for only 32% of everyday questions.
17. **What is an embedding model and which one do you use?** A small neural network that turns text into 384 numbers
    describing its meaning. We use `BAAI/bge-small-en-v1.5` through fastembed (ONNX, no torch, no key, ~70 MB).
18. **Why hybrid and not embeddings only?** Embeddings understand meaning but can mix up similar-looking names
    (our known miss: Trichogramma vs Trichoderma). TF-IDF rewards exact words. The weighted mix (0.7/0.3) uses both.
19. **Why cache the chunk embeddings?** Embedding all 100+ chunks takes time. We save them with a fingerprint of the
    notes; if the notes change, the fingerprint changes and they are rebuilt automatically.
20. **How did you choose the threshold 0.47?** We looked at the best hybrid score of every test question: off-topic
    questions scored at most 0.454 and answerable ones at least 0.487, so we took the middle. The gap is small, and we
    chose it on our own test questions, so it is optimistic.
21. **Why did the grounded score on the original questions go down (0.97 → 0.87)?** The new answer shows only 2-4
    sentences, so sometimes the exact key fact sits in a sentence that was left out. Shorter answers trade a little
    completeness for readability.
22. **Does the app need an LLM or an API key?** No. The default answer copies the best sentences from our notes. If a
    free GROQ_API_KEY is added in Streamlit secrets, an LLM rewrites the answer only from those passages with citations;
    if the key is missing or fails, it falls back automatically.

