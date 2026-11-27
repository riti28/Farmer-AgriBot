"""
Step 4: RAG = Retrieval-Augmented Generation (here: retrieval + extractive answer).

1. Read every note in corpus/ and cut it into small chunks (3 sentences each).
2. Turn every chunk into a TF-IDF vector (important words get bigger numbers).
3. For a question, find the 3 most similar chunks with cosine similarity.
4. Answer ONLY with text from those chunks, and show where it came from.
   If even the best chunk is not similar enough -> "Not in my documents".

Run: python rag.py "How do I control fall armyworm?"
"""
import os
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
TOP_K = 3
THRESHOLD = 0.15  # below this similarity score we say "Not in my documents"
NOT_FOUND = "Not in my documents."


def load_chunks():
    """Return a list of chunks. Each chunk is a dict with text + its source."""
    chunks = []
    for file_name in sorted(os.listdir(CORPUS_DIR)):
        if not file_name.endswith(".txt"):
            continue
        with open(os.path.join(CORPUS_DIR, file_name), encoding="utf-8") as f:
            lines = f.read().split("\n")
        # First 3 lines are the header: "Title: ...", "URL: ...", "Org: ..."
        title = lines[0].replace("Title:", "").strip()
        url = lines[1].replace("URL:", "").strip()
        org = lines[2].replace("Org:", "").strip()
        body = " ".join(lines[3:]).strip()

        sentences = re.split(r"(?<=[.!?])\s+", body)
        for i in range(0, len(sentences), 3):
            chunks.append({
                "doc_id": file_name[:-4],
                "title": title, "url": url, "org": org,
                # put the title in front so the chunk "knows" its topic
                "text": " ".join(sentences[i:i + 3]),
            })
    return chunks


# Build the search index once, when this file is imported
CHUNKS = load_chunks()
VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
MATRIX = VECTORIZER.fit_transform([c["title"] + ". " + c["text"] for c in CHUNKS])


def search(question, k=TOP_K):
    """Return the k chunks most similar to the question, best first."""
    scores = cosine_similarity(VECTORIZER.transform([question]), MATRIX)[0]
    best = scores.argsort()[::-1][:k]
    return [dict(CHUNKS[i], score=round(float(scores[i]), 3)) for i in best]


def answer(question):
    """Answer using ONLY the top-3 chunks. Returns (answer_text, sources)."""
    hits = [h for h in search(question) if h["score"] >= THRESHOLD]
    if not hits:
        return NOT_FOUND, []
    # Extractive answer: the retrieved text itself, each part tagged [1], [2], [3]
    parts = [f"{h['text']} [{n}]" for n, h in enumerate(hits, start=1)]
    return "\n\n".join(parts), hits


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I manage fall armyworm in maize?"
    text, sources = answer(question)
    print("Q:", question, "\n")
    print(text, "\n")
    for n, s in enumerate(sources, start=1):
        print(f"[{n}] {s['title']} - {s['org']} ({s['url']})  score={s['score']}")
