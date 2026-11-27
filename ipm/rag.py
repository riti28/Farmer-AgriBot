"""
Step 4: RAG = Retrieval-Augmented Generation for the "Ask IPM" chat.

1. Read every note in corpus/ and cut it into small chunks (3 sentences each).
2. Expand farmer words with a SYNONYMS list ("worms" -> also "larvae caterpillar").
3. HYBRID SEARCH: score every chunk two ways and mix the scores:
     - TF-IDF  : matches the same WORDS            (fast, exact)
     - Embedding: matches the same MEANING          (BAAI/bge-small-en-v1.5 via fastembed,
                                                     small ONNX model, no torch, no API key)
4. Answer:
     - normal (no key)  : the 2-4 sentences from the top-3 chunks most similar to the
                          question, each tagged [1] [2] [3] with its source
     - optional LLM     : if a GROQ_API_KEY is set, an LLM rewrites the answer ONLY
                          from those passages (still with citations)
     - nothing matches  : "Not in my documents." + 3 closest topics + "Ask your local KVK"

Run: python rag.py "worms are eating my maize"
"""
import hashlib
import os
import re
import sys
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
CACHE_FILE = os.path.join(HERE, "models", "chunk_embeddings.npz")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 3
W_EMBED, W_TFIDF = 0.7, 0.3   # final score = 0.7 * meaning score + 0.3 * word score
THRESHOLD = 0.47              # hybrid score below this -> "Not in my documents" (tuned in eval.py)
TFIDF_THRESHOLD = 0.15        # threshold of the OLD TF-IDF-only version (kept for comparison)
NOT_FOUND = "Not in my documents."
GENERAL_LABEL = "⚠️ Not from our documents – general info, confirm with your KVK"

# Everyday farmer words -> words our notes use. Added to the question before searching.
SYNONYMS = {
    "worm": "larvae caterpillar", "worms": "larvae caterpillar", "caterpillar": "larvae",
    "spray": "insecticide pesticide", "spraying": "insecticide pesticide",
    "medicine": "pesticide insecticide", "chemical": "pesticide insecticide",
    "insect": "pest", "insects": "pest", "bug": "pest", "bugs": "pest",
    "ragi": "finger millet", "paddy": "rice", "corn": "maize", "cane": "sugarcane",
    "spots": "lesions blast disease", "spot": "lesions disease",
    "yellow": "yellowing deficiency nutrient hopper",
    "dying": "wilting damage root rot", "drying": "wilting damage", "dried": "wilting damage",
    "wilt": "wilting", "holes": "caterpillar feeding damage",
    "white ants": "termites", "rat": "rodents", "rats": "rodents", "mice": "rodents",
    "lice": "aphids", "sticky": "honeydew aphids whitefly",
    "kill": "control management", "get rid": "control management",
    "organic": "neem bio-pesticide natural enemies", "natural": "neem natural enemies",
    "fungus": "fungal disease", "dose": "label dose per acre", "how much": "dose per acre",
}


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
        sentences = split_sentences(" ".join(lines[3:]).strip())
        for i in range(0, len(sentences), 3):
            chunks.append({"doc_id": file_name[:-4], "title": title, "url": url, "org": org,
                           "text": " ".join(sentences[i:i + 3])})
    return chunks


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def expand_question(question):
    """Add synonyms of farmer words: 'worms eating maize' -> '... larvae caterpillar'."""
    q = question.lower()
    extra = [words for key, words in SYNONYMS.items() if re.search(r"\b" + key + r"\b", q)]
    return question + " " + " ".join(extra) if extra else question


# ---- Build the TF-IDF index once (title is added so each chunk "knows" its topic) ----
CHUNKS = load_chunks()
CHUNK_TEXTS = [c["title"] + ". " + c["text"] for c in CHUNKS]
VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
MATRIX = VECTORIZER.fit_transform(CHUNK_TEXTS)

# ---- Embedding model: loaded only when first needed (app.py loads it with st.cache_resource) ----
EMBEDDER = None
CHUNK_VECTORS = None


def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        from fastembed import TextEmbedding  # imported here so TF-IDF still works without it
        EMBEDDER = TextEmbedding(MODEL_NAME)  # downloads ~70 MB the first time, then cached
    return EMBEDDER


def get_chunk_vectors():
    """Embeddings of all chunks. Saved to a file and rebuilt only if the notes change."""
    global CHUNK_VECTORS
    if CHUNK_VECTORS is None:
        fingerprint = hashlib.md5((MODEL_NAME + "".join(CHUNK_TEXTS)).encode()).hexdigest()
        if os.path.exists(CACHE_FILE):
            saved = np.load(CACHE_FILE)
            if str(saved["fingerprint"]) == fingerprint:
                CHUNK_VECTORS = saved["vectors"]
        if CHUNK_VECTORS is None:  # no cache, or the notes changed -> rebuild
            CHUNK_VECTORS = np.array(list(get_embedder().embed(CHUNK_TEXTS)))
            np.savez(CACHE_FILE, vectors=CHUNK_VECTORS, fingerprint=fingerprint)
    return CHUNK_VECTORS


def embed_query(text):
    return np.array(list(get_embedder().query_embed([text])))[0]


def search(question, k=TOP_K, method="hybrid", use_synonyms=True, only_docs=None):
    """Return the k best chunks, best first. method = "hybrid" or "tfidf" (old version).
    only_docs = optional set of doc ids to search in (used by eval.py)."""
    query = expand_question(question) if use_synonyms else question
    word_scores = cosine_similarity(VECTORIZER.transform([query]), MATRIX)[0]
    if method == "tfidf":
        scores = word_scores
    else:
        meaning_scores = get_chunk_vectors() @ embed_query(query)  # vectors are normalised
        scores = W_EMBED * meaning_scores + W_TFIDF * word_scores
    if only_docs is not None:
        scores = np.where([c["doc_id"] in only_docs for c in CHUNKS], scores, -1)
    best = scores.argsort()[::-1][:k]
    return [dict(CHUNKS[i], score=round(float(scores[i]), 3)) for i in best]


def best_sentences(question, hits, max_n=4, min_n=2):
    """From the retrieved chunks keep only the 2-4 sentences closest to the question.
    Returns a list of (sentence, source_number)."""
    candidates = [(s, n) for n, hit in enumerate(hits, start=1) for s in split_sentences(hit["text"])]
    chunk_scores = np.array([hits[n - 1]["score"] for _, n in candidates])
    query = expand_question(question)
    sentence_vectors = np.array(list(get_embedder().embed([s for s, _ in candidates])))
    word_scores = cosine_similarity(VECTORIZER.transform([query]),
                                    VECTORIZER.transform([s for s, _ in candidates]))[0]
    sentence_scores = W_EMBED * (sentence_vectors @ embed_query(query)) + W_TFIDF * word_scores
    # half the sentence's own score + half its chunk's score (a sentence alone has little context)
    scores = 0.5 * sentence_scores + 0.5 * chunk_scores
    order = scores.argsort()[::-1]
    keep = [i for i in order[:max_n] if scores[i] >= 0.85 * scores[order[0]]]
    keep = keep if len(keep) >= min_n else list(order[:min_n])
    return [candidates[i] for i in sorted(keep)]  # keep the original reading order


# ---------------- Optional LLM (only if a GROQ_API_KEY is set) ----------------
def get_secret(name):
    """Read a setting from an environment variable or Streamlit secrets. None if missing."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:  # no secrets file, or not running inside Streamlit
        return None


def ask_llm(question, hits):
    """Ask a Groq-hosted LLM. With hits: answer ONLY from them. Without: short general info.
    Returns the answer text, or None if anything goes wrong (then we fall back)."""
    key = get_secret("GROQ_API_KEY")
    if not key:
        return None
    if hits:
        passages = "\n".join(f"[{n}] {h['text']}" for n, h in enumerate(hits, start=1))
        task = ("Answer the farmer's question using ONLY the numbered passages below. "
                "Cite every fact like [1] or [2]. If the passages do not answer it, reply exactly "
                f"'{NOT_FOUND}'.\n\nPassages:\n{passages}")
    else:
        task = ("None of our documents match. If the question is about crops, pests or farming, "
                "give a short, careful general answer (no pesticide doses). Otherwise say you can "
                "only help with crop pest questions.")
    try:
        reply = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": get_secret("GROQ_MODEL") or "llama-3.1-8b-instant", "temperature": 0.2,
                  "messages": [{"role": "system", "content": "You are an IPM assistant for farmers. "
                                "Use simple English and at most 5 sentences. " + task},
                               {"role": "user", "content": question}]},
            timeout=30)
        reply.raise_for_status()
        return reply.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def answer(question):
    """Main function used by the app, eval.py and the MCP server. Returns a dict:
    text, sources (list of chunks), found (True/False), suggestions (topic titles), mode."""
    hits = search(question)
    good = [h for h in hits if h["score"] >= THRESHOLD]

    if not good:  # ---- nothing close enough ----
        suggestions = []
        for hit in search(question, k=10):
            if hit["title"] not in suggestions:
                suggestions.append(hit["title"])
        llm_text = ask_llm(question, [])
        if llm_text:
            return {"text": f"{GENERAL_LABEL}\n\n{llm_text}", "sources": [], "found": False,
                    "suggestions": suggestions[:3], "mode": "llm_general"}
        return {"text": f"{NOT_FOUND} Ask your local KVK or agriculture officer.", "sources": [],
                "found": False, "suggestions": suggestions[:3], "mode": "not_found"}

    llm_text = ask_llm(question, good)
    if llm_text and llm_text.strip() != NOT_FOUND:
        return {"text": llm_text, "sources": good, "found": True, "suggestions": [], "mode": "llm"}

    # ---- no key (or LLM failed): show the best 2-4 sentences with citations ----
    text = " ".join(f"{s} [{n}]" for s, n in best_sentences(question, good))
    return {"text": text, "sources": good, "found": True, "suggestions": [], "mode": "extractive"}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "worms are eating my maize"
    result = answer(question)
    print("Q:", question, f"  (mode: {result['mode']})\n")
    print(result["text"], "\n")
    for n, s in enumerate(result["sources"], start=1):
        print(f"[{n}] {s['title']} - {s['org']} ({s['url']})  score={s['score']}")
    if result["suggestions"]:
        print("Topics we cover:", " | ".join(result["suggestions"]))
