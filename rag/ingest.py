# rag/ingest.py
import os, json, shutil, csv
import ollama

DB_PATH = "rag/db"
EMBED_MODEL = "nomic-embed-text"

# ─── Format extractors (private, used internally by ingest_path) ─────────────

def _extract_pdf(filepath: str) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)

def _extract_csv(filepath: str) -> str:
    lines = []
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lines.append(", ".join(f"{k}: {v}" for k, v in row.items() if v.strip()))
    return "\n".join(lines)

def _extract_txt(filepath: str) -> str:
    with open(filepath, encoding='utf-8') as f:
        return f.read()

_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".csv": _extract_csv,
    ".txt": _extract_txt,
    ".md":  _extract_txt,
}

# ─── Core embedding + storage ─────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]

def ingest_document(doc_id: str, content: str, metadata: dict = None):
    """Embed raw text and store it. Used directly for manual/programmatic entries."""
    os.makedirs(DB_PATH, exist_ok=True)
    vector = embed_text(content)
    entry = {"id": doc_id, "content": content, "metadata": metadata or {}, "vector": vector}
    with open(os.path.join(DB_PATH, f"{doc_id}.json"), "w") as f:
        json.dump(entry, f)
    print(f"[RAG] ✅ Ingested: '{doc_id}' ({len(content)} chars)")

def ingest_path(filepath: str, doc_id: str = None, metadata: dict = None):
    """
    Single entry point for ANY file type. Auto-detects format, extracts text,
    embeds, and stores. This is the one function you call from now on.
    """
    ext = os.path.splitext(filepath)[1].lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        print(f"[RAG] ⚠️  Unsupported format '{ext}', skipping: {filepath}")
        return

    doc_id = doc_id or os.path.splitext(os.path.basename(filepath))[0]
    print(f"[RAG] 📂 Loading: {filepath}")
    content = extractor(filepath)

    if not content.strip():
        print(f"[RAG] ⚠️  Empty content, skipping: {filepath}")
        return

    meta = metadata or {}
    meta.setdefault("source", filepath)
    ingest_document(doc_id, content, meta)

def ingest_folder(folder: str = "data"):
    """Ingest every supported file in a folder in one call. Simplest workflow."""
    if not os.path.isdir(folder):
        print(f"[RAG] ⚠️  Folder not found: {folder}")
        return
    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in _EXTRACTORS:
            ingest_path(os.path.join(folder, filename))

# ─── Management ────────────────────────────────────────────────────────────

def update_document(doc_id: str, content: str, metadata: dict = None):
    print(f"[RAG] 🔄 Updating: '{doc_id}'")
    ingest_document(doc_id, content, metadata)

def delete_document(doc_id: str):
    filepath = os.path.join(DB_PATH, f"{doc_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[RAG] 🗑️  Deleted: '{doc_id}'")
    else:
        print(f"[RAG] ⚠️  Not found: '{doc_id}'")

def flush_all():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("[RAG] 💥 Knowledge base flushed.")

def list_documents() -> list[str]:
    if not os.path.exists(DB_PATH):
        return []
    return [f.replace(".json", "") for f in os.listdir(DB_PATH) if f.endswith(".json")]

if __name__ == "__main__":
    print("Ingesting everything in data/ ...\n")
    ingest_folder("data")
    print(f"\nDocuments stored: {list_documents()}")