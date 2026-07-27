# Personal Assistant

A local, LLM-centered personal assistant built for eventual deployment on a
Raspberry Pi 5. It uses lightweight local models via Ollama and custom Python
adapters (no LangChain, no n8n) so the whole stack stays lean enough to run
fully on-device.

## Status

Working end-to-end: agentic tool-calling (Ollama native tool calls) + a local
RAG store (file ingestion, embedding, retrieval) are both functional and
tested on a laptop with an RTX 4060, using `qwen2.5:3b` as the routing model
and `nomic-embed-text` for embeddings. Real API integrations (Gmail,
Google Calendar) are the next milestone — everything up to that point is
dummy/mocked and safe to run offline.

## Setup

### 1. Install Ollama and pull the required models

```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Both models must be identical across every machine that touches this repo
(laptop and Pi). The RAG database stores embedding vectors — if the
embedding model differs between machines, previously-ingested vectors won't
match new queries and retrieval breaks silently.

### 2. Clone and set up a virtual environment

```bash
git clone <repo-url>
cd Personal-Assistant

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux/Pi:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Verify Ollama is running

```bash
ollama list   # should show qwen2.5:3b and nomic-embed-text
```

### 4. Run the orchestrator

```bash
python orchestrate.py
```

This runs a sample query through the full tool-calling pipeline and prints
diagnostics (which tool was called, arguments, raw tool output, and the
model's final natural-language response).

## Repo structure

```
Personal-Assistant/
├── orchestrate.py       # Entry point. Defines AgentRequest/AgentResponse,
│                         the OllamaTransport, the tool registry, and the
│                         Orchestrator class (retry loop + tool dispatch +
│                         final response generation).
├── requirements.txt
├── README.md
├── tools/                # One file per agentic capability. Each tool is a
│  ├── __init__.py         plain Python function with a docstring — the
│  ├── read_email.py       docstring IS the tool schema the model sees, so
│  ├── get_calendar_events.py   keep argument descriptions explicit and
│  ├── add_calendar_event.py    concrete (this directly affects whether the
│  ├── draft_email.py           model fills arguments correctly).
│  └── query_rag.py        Currently dummy/mocked; will be swapped for real
│                          Gmail/Calendar API calls without changing their
│                          function signatures (orchestrate.py won't need
│                          to change when that happens).
├── rag/                  # Local retrieval-augmented generation store.
│  ├── __init__.py
│  ├── ingest.py           Single entry point for all ingestion:
│  │                         - ingest_document(id, text, metadata) — raw text
│  │                         - ingest_path(filepath) — one file, any format
│  │                         - ingest_folder("data") — ingest everything in
│  │                           data/ at once (the normal workflow)
│  │                         - update_document / delete_document / flush_all
│  │                         Supports .pdf, .csv, .txt, .md out of the box;
│  │                         add a new format by adding one entry to
│  │                         _EXTRACTORS.
│  ├── retriever.py         retrieve(query, top_k) — cosine-similarity search
│  │                         over the local vector store. format_context()
│  │                         turns results into a string for the LLM prompt.
│  └── db/                  Auto-generated. One JSON file per ingested
│                            document (id, content, metadata, embedding
│                            vector). Don't edit by hand — use ingest.py's
│                            functions instead. Safe to delete entirely to
│                            reset the knowledge base (or call flush_all()).
└── data/                 # Drop source files here (PDF/CSV/TXT/MD) then run
                            `python rag/ingest.py` to ingest everything in
                            this folder into the RAG store.
```

## Day-to-day workflows

**Add new knowledge to the assistant's memory:**

```bash
# 1. Drop a file into data/
# 2. Ingest everything in data/
python rag/ingest.py
```

**Test retrieval on its own (no LLM orchestration):**

```bash
python rag/retriever.py
```

**Update or remove a specific document:**

```python
from rag.ingest import update_document, delete_document
update_document("course_ml", "New content here...", {"type": "course"})
delete_document("old_doc_id")
```

**Reset the knowledge base entirely:**

```python
from rag.ingest import flush_all
flush_all()
```

**Run the full agentic pipeline:**

```bash
python orchestrate.py
```

## Architecture notes

- **Two-tier context**: slow-changing facts (courses, preferences, project
  status) live in the RAG store. Fast-changing live data (calendar events,
  emails) is fetched fresh via tools at request time and passed into the
  same prompt alongside RAG results. Tool results that represent new
  standing facts (e.g. a new task from an email) should be written back into
  the RAG store via `update_document`, so the assistant's long-term memory
  stays current.
- **Model choice**: `qwen2.5:1.5b` was tested first and needed 3-4 retries
  per tool call to reliably trigger tool use. `qwen2.5:3b` gets it right on
  the first try in testing so far and is the current default. Same model
  must run on the Pi 5 for consistent behavior.
- **LLMTransport protocol**: `orchestrate.py` defines transport as a
  protocol/interface so the underlying model runtime (currently Ollama) can
  be swapped later without touching the Orchestrator or tool logic.
- **Retry loop**: if the model replies with plain text instead of a tool
  call, the orchestrator appends a correction message and retries, up to
  `max_retries`. Every attempt is logged, including any reasoning text the
  model produced, for diagnosing empty or incorrect tool arguments.

## Known gaps / next milestones

1. **Fix empty-argument cases** — some tool calls (e.g. `add_calendar_event`)
   have returned empty arguments in testing. Next step is tightening tool
   docstrings so required arguments are explicit, and re-testing.
2. **Chunking long documents** — currently a whole ingested file becomes one
   embedding vector. Fine for short text, but long PDFs will need to be
   split into smaller chunks before embedding for retrieval quality to hold
   up. Not yet implemented.
3. **Real API integration** — swap `tools/read_email.py`,
   `tools/draft_email.py`, `tools/get_calendar_events.py`, and
   `tools/add_calendar_event.py` from mocked responses to real Gmail and
   Google Calendar API calls (OAuth2). Function signatures should stay the
   same so `orchestrate.py` doesn't need changes.
4. **Sensor integration** — camera (face/gaze detection via MediaPipe) and
   microphone (wake word + speaker recognition + Whisper STT) are not yet
   wired into this repo. These will feed into the same orchestrator as
   additional context/triggers.
5. **Raspberry Pi 5 deployment** — once the above are stable on the laptop,
   `git pull` this repo on the Pi, recreate the virtual environment, pull
   the same Ollama models, and validate the same test cases. Expect higher
   per-call latency than the RTX 4060 (no discrete GPU); the current
   design avoids running LLM inference concurrently with camera/mic
   inference to stay within the Pi's memory budget.

## For teammates joining this repo

- Don't call real Gmail/Calendar APIs yet — everything is mocked
  intentionally so the agentic + RAG layers can be tested without OAuth
  setup or API quota concerns.
- If you add a new tool, put it in its own file under `tools/`, register it
  in `TOOL_REGISTRY` in `orchestrate.py`, and write a precise docstring —
  that docstring is what the model uses to decide when and how to call it.
- If you add support for a new file format to the RAG store, add one entry
  to `_EXTRACTORS` in `rag/ingest.py` — no other changes needed.
- Keep `qwen2.5:3b` and `nomic-embed-text` as the pinned models until a
  deliberate decision is made to change them (changing the embedding model
  invalidates the existing `rag/db/` contents).

## Model Comparison

- **Qwen2.5:3B**: 2.5 mins to run orchestrate.py, 100% success every tool call. MAX ram usage 4.5gb first tool fetch then plateud at 3gb usage with vscode ssh.

- **Qwen2.5:1.5B**: 5 mins to run orchestrate.py, failed 3 tool calls 10 times in a row and timed out. MAX ram usage 2gb with vscode ssh.
