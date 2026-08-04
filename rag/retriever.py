# rag/retriever.py
import os, json
import ollama
import numpy as np

DB_PATH = "rag/db"
EMBED_MODEL = "nomic-embed-text"

def embed_text(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _load_db() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    entries = []
    for filename in os.listdir(DB_PATH):
        if filename.endswith(".json"):
            with open(os.path.join(DB_PATH, filename)) as f:
                entries.append(json.load(f))
    return entries

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_vector = embed_text(query)
    db = _load_db()
    if not db:
        return []
    scored = [
        {"id": e["id"], "content": e["content"], "metadata": e["metadata"],
         "score": _cosine_similarity(query_vector, e["vector"])}
        for e in db
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

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