"""
Step 5: Evaluate the RAG search on 30 test questions.

Hit@3          = for answerable questions: is a correct document in the top 3 results?
Grounded score = is the answer right AND built only from documents?
                 - answerable question: answer cites an expected document AND contains
                   the key fact (answer_keyword)
                 - out-of-scope question (expected "none"): answer is "Not in my documents"

Output: eval/results.md
Run:    python eval.py
"""
import os
import pandas as pd
import rag

HERE = os.path.dirname(os.path.abspath(__file__))
questions = pd.read_csv(os.path.join(HERE, "eval", "questions.csv"), keep_default_na=False)

rows = []
for _, q in questions.iterrows():
    expected = q["expected_doc_ids"].split("|")
    top3 = [hit["doc_id"] for hit in rag.search(q["question"])]
    text, sources = rag.answer(q["question"])
    cited = [s["doc_id"] for s in sources]

    if expected == ["none"]:
        hit = None  # Hit@3 only makes sense for answerable questions
        grounded = text == rag.NOT_FOUND
    else:
        hit = any(doc in expected for doc in top3)
        grounded = any(doc in expected for doc in cited) and q["answer_keyword"].lower() in text.lower()
    rows.append({"id": q["id"], "question": q["question"], "top3": ", ".join(top3),
                 "hit@3": hit, "grounded": grounded})

res = pd.DataFrame(rows)
answerable = res[res["hit@3"].notna()]
hit_at_3 = answerable["hit@3"].astype(bool).mean()
grounded_score = res["grounded"].mean()
print(f"Hit@3 = {hit_at_3:.2f} ({int(answerable['hit@3'].sum())}/{len(answerable)} answerable questions)")
print(f"Grounded-answer score = {grounded_score:.2f} ({int(res['grounded'].sum())}/{len(res)} questions)")

with open(os.path.join(HERE, "eval", "results.md"), "w", encoding="utf-8") as f:
    f.write("# RAG evaluation results\n\n")
    f.write(f"- Retriever: TF-IDF + cosine similarity, top-{rag.TOP_K}, threshold {rag.THRESHOLD}\n")
    f.write(f"- Corpus: {len(set(c['doc_id'] for c in rag.CHUNKS))} notes, {len(rag.CHUNKS)} chunks\n")
    f.write("- Note: we wrote these questions from our own notes and picked the threshold by looking at\n"
            "  them, so the scores are optimistic. Real farmer questions would score lower.\n\n")
    f.write("| Metric | Score |\n|---|---|\n")
    f.write(f"| Hit@3 ({len(answerable)} answerable questions) | {hit_at_3:.2f} |\n")
    f.write(f"| Grounded-answer score (all {len(res)} questions) | {grounded_score:.2f} |\n\n")
    f.write("## Per-question results\n\n")
    f.write("| id | question | top-3 doc ids | hit@3 | grounded |\n|---|---|---|---|---|\n")
    for _, r in res.iterrows():
        hit = "-" if r["hit@3"] is None else r["hit@3"]
        f.write(f"| {r['id']} | {r['question']} | {r['top3']} | {hit} | {r['grounded']} |\n")
print("Saved eval/results.md")
