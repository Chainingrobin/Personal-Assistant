# Personal Assistant

A local, LLM-centered personal assistant built for a Raspberry Pi 5 target. The runtime stays deliberately simple: Ollama for routing, plain Python tool functions for side effects, and direct Google API calls for email and calendar access.

## Setup

### 1. Install Ollama and pull the models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 2. Create the Google Cloud project

- Create one Google Cloud project for this assistant.
- Enable the Gmail API and Google Calendar API.
- Create one OAuth Desktop client.
- Download the client secret JSON and save it as `credentials/credentials.json`.

### 3. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Enroll each user once on a laptop

- Run the assistant on a machine with a browser.
- Trigger a Google-backed tool for that user once.
- Complete the consent flow.
- The app writes a per-user token file to `credentials/tokens/<user_id>.json`.
- Copy that token file to the Raspberry Pi for the same user.

The Pi should not need to open a browser for OAuth. In normal deployment, tokens already exist before the Pi runs the assistant.

### 5. Run the orchestrator

```bash
python orchestrate.py
```

## How it is organized

```
Personal-Assistant/
├── config.py            # Root config for Ollama and Google settings.
├── credentials/         # Local-only OAuth material.
│  ├── credentials.json  # Google OAuth Desktop client secret.
│  └── tokens/           # Per-user token files: <user_id>.json.
├── orchestrate.py       # Tool routing, retry loop, and tool dispatch.
├── requirements.txt
├── tools/
│  ├── auth.py           # Google OAuth token loading/refresh/consent.
│  ├── read_email.py     # Gmail API inbox reads.
│  ├── get_calendar_events.py  # Calendar API event lookup.
│  ├── add_calendar_event.py    # Calendar API event insert.
│  ├── draft_email.py    # Gmail API draft creation only.
│  └── query_rag.py      # Local RAG lookup; unchanged.
├── rag/
└── data/
```

## Runtime model

- `user_id` is injected by the orchestrator after the model picks a tool.
- The model handles tool selection and task arguments only.
- User identity comes from the face/voice layer elsewhere in the system.
- Google OAuth tokens are stored per user so one person’s account never overwrites another’s.
- The laptop is used for first-time consent; the Pi only reuses copied token files.

## Notes

- `config.py` loads `.env` automatically with `python-dotenv`.
- `.env`, `credentials/`, and Google token files are ignored by Git.
- `tools/query_rag.py` remains separate from the Google integrations.
- The old synthetic test harness under `tests/` was removed from this trimmed runtime snapshot.
