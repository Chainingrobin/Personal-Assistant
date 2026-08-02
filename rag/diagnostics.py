# rag/diagnostics.py
"""
Lightweight diagnostics layer for the RAG pipeline.

This file owns ONLY logging/inspection helpers. It does not do embedding,
storage, or retrieval itself — ingest.py and retriever.py import from here
and call these functions at key checkpoints. Keeping this separate means
you can silence, redirect, or upgrade logging (e.g. to a file or JSON log)
without touching the actual RAG logic.

Everything prints to console by default AND appends structured JSON lines
to rag/diagnostics.log, so you have both a human-readable trace during
testing and a machine-readable trail you can review later.
"""

import json
import os
import time
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "diagnostics.log")


def _write_log(record: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_ingest(doc_id: str, filepath: str, ext: str, char_count: int,
               vector_dim: int, embed_seconds: float, error: str | None = None):
    status = "ERROR" if error else "OK"
    record = {
        "event": "ingest",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "doc_id": doc_id,
        "filepath": filepath,
        "ext": ext,
        "char_count": char_count,
        "vector_dim": vector_dim,
        "embed_seconds": round(embed_seconds, 3),
        "status": status,
        "error": error,
    }
    _write_log(record)

    icon = "❌" if error else "✅"
    print(f"[DIAG][INGEST] {icon} id='{doc_id}' file='{filepath}' ext={ext} "
          f"chars={char_count} vector_dim={vector_dim} "
          f"embed_time={embed_seconds:.3f}s status={status}"
          + (f" error='{error}'" if error else ""))

    # Sanity flags a beginner should look at
    if not error:
        if char_count == 0:
            print(f"[DIAG][WARN] '{doc_id}' produced 0 characters of extracted text. "
                  f"Nothing useful will ever be retrieved for this document — check the source file.")
        elif char_count < 20:
            print(f"[DIAG][WARN] '{doc_id}' has suspiciously little text ({char_count} chars). "
                  f"Double check extraction worked for this file type.")
        if vector_dim == 0:
            print(f"[DIAG][WARN] '{doc_id}' got an empty embedding vector. Embedding model may have failed silently.")


def log_retrieve(query: str, top_k: int, scored_results: list[dict],
                  embed_seconds: float, relevance_threshold: float = 0.35):
    record = {
        "event": "retrieve",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "top_k": top_k,
        "embed_seconds": round(embed_seconds, 3),
        "results": [
            {"id": r["id"], "score": round(r["score"], 4)} for r in scored_results
        ],
    }
    _write_log(record)

    print(f"\n[DIAG][RETRIEVE] query='{query}' (embed_time={embed_seconds:.3f}s)")
    if not scored_results:
        print("[DIAG][WARN] No documents in the knowledge base at all — nothing to compare against.")
        return

    best_score = scored_results[0]["score"]
    for rank, r in enumerate(scored_results, start=1):
        flag = ""
        if r["score"] < relevance_threshold:
            flag = "  ⚠️ LOW CONFIDENCE (below threshold)"
        print(f"  #{rank} id='{r['id']}' score={r['score']:.4f}{flag}")

    if best_score < relevance_threshold:
        print(f"[DIAG][WARN] Best match score ({best_score:.4f}) is below the "
              f"relevance threshold ({relevance_threshold}). This usually means "
              f"none of the ingested documents actually answer this query — "
              f"the model may hallucinate an answer from a weak match. Consider "
              f"ingesting more relevant content or rephrasing the query.")

    # Detect near-ties, which often mean ambiguous / duplicate content
    if len(scored_results) >= 2:
        gap = scored_results[0]["score"] - scored_results[1]["score"]
        if gap < 0.02:
            print(f"[DIAG][WARN] Top 2 results are nearly tied (gap={gap:.4f}). "
                  f"Check whether these documents overlap in content.")


class Timer:
    """Small context manager so ingest/retrieve code can time a block cleanly."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
