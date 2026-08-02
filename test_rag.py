"""
test_rag.py — end-to-end RAG correctness test using real PDF/CSV/TXT/MD files.

What this does:
  1. Wipes the existing rag/db/ so results are reproducible.
  2. Ingests every file in test_data/ (one file per format: .pdf, .csv, .txt, .md).
  3. Runs a "golden set" of queries where you already know which document
     SHOULD come back on top.
  4. Checks the actual top result against the expected document and prints
     a PASS/FAIL report, plus the diagnostic output from rag/diagnostics.py.

Run this from the project root (same folder as orchestrate.py):
    python test_rag.py

Requirements: Ollama running locally with `nomic-embed-text` pulled.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ingest import ingest_folder, flush_all, list_documents
from rag.retriever import retrieve

DATA_FOLDER = "test_data"  # rename to "data" if you want to test your real folder

# Each entry: (query, expected_doc_id)
# expected_doc_id is the filename without extension, e.g. course_notes.txt -> "course_notes"
GOLDEN_QUERIES = [
    ("When is my linear algebra exam?", "course_notes"),
    ("What are my professor's office hours?", "course_notes"),
    ("Do I prefer coffee or tea?", "preferences"),
    ("Should the IDE use dark mode?", "preferences"),
    ("What tasks are marked high priority?", "tasks"),
    ("When do I need to buy an SD card?", "tasks"),
    ("What is Sprint 2's main goal?", "project_brief"),
    ("Which routing model is currently pinned and why?", "project_brief"),
]


def run():
    print("=" * 70)
    print("STEP 1: Resetting knowledge base")
    print("=" * 70)
    flush_all()

    print("\n" + "=" * 70)
    print(f"STEP 2: Ingesting all files in '{DATA_FOLDER}/'")
    print("=" * 70)
    ingest_folder(DATA_FOLDER)

    stored = list_documents()
    print(f"\nDocuments now in the knowledge base: {stored}")
    expected_ids = {doc_id for _, doc_id in GOLDEN_QUERIES}
    missing = expected_ids - set(stored)
    if missing:
        print(f"[TEST][WARN] These expected documents never made it into the DB: {missing}. "
              f"Check the [DIAG][INGEST] errors above.")

    print("\n" + "=" * 70)
    print("STEP 3: Running golden queries and checking relevance")
    print("=" * 70)

    results_table = []
    for query, expected_id in GOLDEN_QUERIES:
        results = retrieve(query, top_k=3)
        top_id = results[0]["id"] if results else None
        top_score = results[0]["score"] if results else 0.0
        passed = (top_id == expected_id)
        results_table.append((query, expected_id, top_id, top_score, passed))

    print("\n" + "=" * 70)
    print("STEP 4: SUMMARY REPORT")
    print("=" * 70)
    passed_count = 0
    for query, expected_id, top_id, top_score, passed in results_table:
        status = "✅ PASS" if passed else "❌ FAIL"
        if passed:
            passed_count += 1
        print(f"{status} | expected='{expected_id:<15}' got='{str(top_id):<15}' "
              f"score={top_score:.3f} | query: {query}")

    total = len(results_table)
    print(f"\n{passed_count}/{total} queries retrieved the expected document as the top result.")
    if passed_count < total:
        print("Look at the [DIAG][RETRIEVE] blocks above each failing query — check "
              "whether the wrong document scored higher, or whether all scores were "
              "low (a sign the embedding model isn't distinguishing your documents well).")

    print(f"\nFull machine-readable trace written to: rag/diagnostics.log")


if __name__ == "__main__":
    run()
