# rag/retriever.py
import os, json, time
import ollama
import numpy as np

from rag.diagnostics import log_retrieve

DB_PATH = "rag/db"
EMBED_MODEL = "nomic-embed-text"
RELEVANCE_THRESHOLD = 0.35  # tune this after you see real score distributions

def embed_text(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _load_db() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    entries = []
    for filename in os.listdir(DB_PATH):
        if filename.endswith(".json"):
            with open(os.path.join(DB_PATH, filename)) as f:
                entries.append(json.load(f))
    return entries

def retrieve(query: str, top_k: int = 3, *, verbose: bool = True) -> list[dict]:
    start = time.perf_counter()
    query_vector = embed_text(query)
    embed_seconds = time.perf_counter() - start

    db = _load_db()
    if not db:
        if verbose:
            log_retrieve(query, top_k, [], embed_seconds, RELEVANCE_THRESHOLD)
        return []

    scored = [
        {"id": e["id"], "content": e["content"], "metadata": e["metadata"],
         "score": _cosine_similarity(query_vector, e["vector"])}
        for e in db
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored[:top_k]

    if verbose:
        # Log against the FULL scored list so you can see how the top_k
        # compares to everything else in the knowledge base, not just the
        # winners — useful for spotting "everything scores about the same"
        # problems, which usually mean bad/degenerate embeddings.
        log_retrieve(query, top_k, scored, embed_seconds, RELEVANCE_THRESHOLD)

    return top_results

def format_context(results: list[dict]) -> str:
    if not results:
        return "No relevant context found."
    return "\n".join(
        f"[{r['metadata'].get('type', 'info').upper()}] {r['content']}"
        for r in results
    )

if __name__ == "__main__":
    for q in ["What are my highest priority tasks?", "When should I take a break?"]:
        print(f"\nQuery: '{q}'")
        results = retrieve(q, top_k=2)
        print(format_context(results))
        print(f"(scores: {[round(r['score'], 3) for r in results]})")
