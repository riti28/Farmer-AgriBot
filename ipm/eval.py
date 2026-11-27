"""
Step 5: Evaluate the "Ask IPM" search on two question sets and compare OLD vs NEW.

Question sets:
  eval/questions.csv         - 30 questions written in the words of our notes
  eval/farmer_questions.csv  - 28 everyday farmer questions ("worms eating my maize")
Each set has a few off-topic questions (expected "none") that should be refused.

Systems compared:
  OLD             : TF-IDF only, original 29 notes, threshold 0.15, answer = whole chunks
  OLD + new notes : same, but with all 49 notes (shows what the new notes alone add)
  NEW             : hybrid search (embeddings + TF-IDF) + synonyms + 49 notes + 2-4 best sentences

Metrics:
  Hit@3    = answerable questions where a correct note is in the top 3 results
  Answered = answerable questions that got an answer (not "Not in my documents")
  Grounded = answer cites a correct note AND contains the key fact (answer_keyword, "|" = any of),
             or, for an off-topic question, the bot correctly refuses
No LLM is used here (the eval always runs the free no-key path).

Output: eval/results.md      Run: python eval.py
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import rag

os.environ.pop("GROQ_API_KEY", None)  # always test the no-key path
HERE = os.path.dirname(os.path.abspath(__file__))
OLD_DOCS = {c["doc_id"] for c in rag.CHUNKS if int(c["doc_id"][3:5]) <= 29}  # ipm01 - ipm29
QUESTION_SETS = {"Original questions": "questions.csv", "Farmer questions": "farmer_questions.csv"}


def make_old_search(only_docs):
    """The OLD search: its own TF-IDF index built only from the chosen notes, no synonyms."""
    chunks = [c for c in rag.CHUNKS if only_docs is None or c["doc_id"] in only_docs]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([c["title"] + ". " + c["text"] for c in chunks])

    def search(question):
        scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
        return [dict(chunks[i], score=float(scores[i])) for i in scores.argsort()[::-1][:3]]
    return search


OLD_SEARCH = {"29 notes": make_old_search(OLD_DOCS), "49 notes": make_old_search(None)}


def old_answer(question, notes):
    """The OLD version: TF-IDF only, no synonyms, answer = the whole top chunks."""
    hits = OLD_SEARCH[notes](question)
    good = [h for h in hits if h["score"] >= rag.TFIDF_THRESHOLD]
    return {"text": " ".join(h["text"] for h in good) or rag.NOT_FOUND, "sources": good,
            "found": bool(good), "top3": [h["doc_id"] for h in hits]}


def new_answer(question):
    result = rag.answer(question)
    result["top3"] = [h["doc_id"] for h in rag.search(question)]
    return result


SYSTEMS = {
    "OLD: TF-IDF, 29 notes": lambda q: old_answer(q, "29 notes"),
    "OLD + new notes: TF-IDF, 49 notes": lambda q: old_answer(q, "49 notes"),
    "NEW: hybrid + synonyms, 49 notes": new_answer,
}


def score_question(row, result):
    expected = row["expected_doc_ids"].split("|")
    if expected == ["none"]:
        return {"hit": None, "answered": None, "grounded": not result["found"]}
    cited = [s["doc_id"] for s in result["sources"]]
    has_fact = any(word.strip().lower() in result["text"].lower()
                   for word in row["answer_keyword"].split("|"))
    return {"hit": any(d in expected for d in result["top3"]),
            "answered": result["found"],
            "grounded": result["found"] and any(d in expected for d in cited) and has_fact}


summary, details = [], []
for set_name, file_name in QUESTION_SETS.items():
    questions = pd.read_csv(os.path.join(HERE, "eval", file_name), keep_default_na=False)
    for system_name, run in SYSTEMS.items():
        rows = []
        for _, q in questions.iterrows():
            result = run(q["question"])
            rows.append(dict(score_question(q, result), question=q["question"], text=result["text"],
                             top3=", ".join(result["top3"])))
        res = pd.DataFrame(rows)
        answerable = res[res["hit"].notna()]
        off_topic = res[res["hit"].isna()]
        summary.append({
            "Question set": f"{set_name} ({len(answerable)} + {len(off_topic)} off-topic)",
            "System": system_name,
            "Hit@3": round(answerable["hit"].astype(bool).mean(), 2),
            "Answered": round(answerable["answered"].astype(bool).mean(), 2),
            "Off-topic refused": f"{int(off_topic['grounded'].sum())}/{len(off_topic)}",
            "Grounded (all)": round(res["grounded"].mean(), 2),
        })
        if system_name.startswith("NEW"):
            details.append((set_name, res))

table = pd.DataFrame(summary)
print(table.to_string(index=False))


# ---- Threshold check: hybrid top score of every question (both sets) ----
def top_scores(file_name):
    q = pd.read_csv(os.path.join(HERE, "eval", file_name), keep_default_na=False)
    return [(rag.search(t)[0]["score"], e == "none") for t, e in zip(q["question"], q["expected_doc_ids"])]


all_scores = top_scores("questions.csv") + top_scores("farmer_questions.csv")
sweep = []
for t in [0.40, 0.45, 0.46, 0.47, 0.48, 0.50, 0.55, 0.60]:
    sweep.append({"threshold": t,
                  "answerable questions answered": sum(s >= t for s, off in all_scores if not off),
                  "off-topic questions refused": sum(s < t for s, off in all_scores if off)})
sweep = pd.DataFrame(sweep)
n_in = sum(not off for _, off in all_scores)
n_off = sum(off for _, off in all_scores)


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


with open(os.path.join(HERE, "eval", "results.md"), "w", encoding="utf-8") as f:
    f.write("# Ask IPM evaluation: OLD vs NEW\n\n")
    f.write(f"- NEW: {rag.MODEL_NAME} embeddings (weight {rag.W_EMBED}) + TF-IDF (weight {rag.W_TFIDF}), "
            f"synonyms, threshold {rag.THRESHOLD}, {len({c['doc_id'] for c in rag.CHUNKS})} notes, "
            f"{len(rag.CHUNKS)} chunks. No LLM.\n")
    f.write(f"- OLD: TF-IDF only, threshold {rag.TFIDF_THRESHOLD}.\n")
    f.write("- Honest caveats: we wrote both question sets ourselves, the new notes were written with\n"
            "  everyday farmer problems in mind, and the threshold was chosen by looking at these same\n"
            "  questions. Real farmer questions would score lower. 'Grounded' only checks one key word.\n\n")
    f.write("## Summary\n\n" + md_table(table))
    f.write(f"\n## Threshold check (NEW system, {n_in} answerable + {n_off} off-topic questions)\n\n")
    f.write(md_table(sweep))
    for set_name, res in details:
        f.write(f"\n## NEW system, per question: {set_name}\n\n")
        out = res[["question", "top3", "hit", "answered", "grounded"]].copy()
        out["hit"] = out["hit"].map(lambda v: "-" if v is None else v)
        out["answered"] = out["answered"].map(lambda v: "-" if v is None else v)
        f.write(md_table(out))
print(sweep.to_string(index=False))
print("Saved eval/results.md")
