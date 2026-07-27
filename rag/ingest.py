import os
import json
import ollama
import numpy as np

# Where we store the vector database (simple flat files for now)
DB_PATH = "rag/db"
EMBED_MODEL = "nomic-embed-text"

def embed_text(text: str) -> list[float]:
    """Get embedding vector for a piece of text."""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]

def ingest_document(doc_id: str, content: str, metadata: dict = None):
    """
    Embed a document and save it to the vector store.
    
    Args:
        doc_id: Unique identifier (e.g. "course_ml", "project_jarvis")
        content: The text to embed and store
        metadata: Optional dict with extra info (priority, type, etc.)
    """
    os.makedirs(DB_PATH, exist_ok=True)
    
    vector = embed_text(content)
    
    entry = {
        "id": doc_id,
        "content": content,
        "metadata": metadata or {},
        "vector": vector,
    }
    
    filepath = os.path.join(DB_PATH, f"{doc_id}.json")
    with open(filepath, "w") as f:
        json.dump(entry, f)
    
    print(f"[RAG] ✅ Ingested: '{doc_id}' ({len(content)} chars)")

def ingest_all_defaults():
    """
    Ingest a set of default student profile documents.
    Run this once to populate the knowledge base.
    """
    documents = [
        {
            "id": "course_ml",
            "content": "Advanced Machine Learning — Assignment due in 3 days, worth 20% of grade. Professor emails weekly updates.",
            "metadata": {"type": "course", "priority": "high"},
        },
        {
            "id": "course_embedded",
            "content": "Embedded Systems project — Jarvis AI assistant on Raspberry Pi 5. Demo in 2 weeks. Group project with 3 members.",
            "metadata": {"type": "project", "priority": "high"},
        },
        {
            "id": "preference_study",
            "content": "User prefers studying in 90-minute focused blocks with 15-minute breaks. Dislikes back-to-back meetings. Best focus time is morning.",
            "metadata": {"type": "preference", "priority": "low"},
        },
        {
            "id": "preference_breaks",
            "content": "For breaks, user enjoys retro gaming or a short walk. Prefers not to be interrupted during active study blocks.",
            "metadata": {"type": "preference", "priority": "low"},
        },
        {
            "id": "internship_status",
            "content": "Currently applying for summer internships in embedded systems or ML. Has one interview scheduled next week with a robotics company.",
            "metadata": {"type": "internship", "priority": "medium"},
        },
    ]
    
    for doc in documents:
        ingest_document(doc["id"], doc["content"], doc["metadata"])

if __name__ == "__main__":
    print("Ingesting default documents into RAG store...\n")
    ingest_all_defaults()
    print("\nDone. Run rag/retriever.py to test queries.")