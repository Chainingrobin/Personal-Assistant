import os
import json
import ollama
import numpy as np

DB_PATH = "rag/db"
EMBED_MODEL = "nomic-embed-text"

def embed_text(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def load_db() -> list[dict]:
    """Load all documents from the flat-file vector store."""
    if not os.path.exists(DB_PATH):
        return []
    entries = []
    for filename in os.listdir(DB_PATH):
        if filename.endswith(".json"):
            with open(os.path.join(DB_PATH, filename)) as f:
                entries.append(json.load(f))
    return entries

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Find the most relevant documents for a query using cosine similarity.
    
    Returns a list of dicts with 'content', 'metadata', and 'score'.
    """
    query_vector = embed_text(query)
    db = load_db()
    
    if not db:
        return []
    
    scored = []
    for entry in db:
        score = cosine_similarity(query_vector, entry["vector"])
        scored.append({
            "id": entry["id"],
            "content": entry["content"],
            "metadata": entry["metadata"],
            "score": score,
        })
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def format_context(results: list[dict]) -> str:
    """Format retrieval results into a string the LLM can read."""
    if not results:
        return "No relevant context found."
    lines = []
    for r in results:
        lines.append(f"[{r['metadata'].get('type', 'info').upper()}] {r['content']}")
    return "\n".join(lines)

if __name__ == "__main__":
    # Quick test — run after ingest.py
    test_queries = [
        "What are my highest priority tasks?",
        "When should I take a break?",
        "What do I have coming up for internships?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retrieve(query, top_k=2)
        print(format_context(results))
        print(f"(scores: {[round(r['score'], 3) for r in results]})")