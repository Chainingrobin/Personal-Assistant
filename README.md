# Personal Assistant

A local, LLM-centered personal assistant built for a Raspberry Pi 5 target. The runtime stays deliberately simple: Ollama for routing, plain Python tool functions for side effects, and direct Google API calls for email and calendar access.

The repo now also includes a separate vision path for study mode: a camera backend abstraction, MediaPipe-based head-pose estimation, and a monitor that can run while the assistant is otherwise idle.

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

If you are deploying to the Raspberry Pi camera module, also install the Pi-only camera extras:

```bash
pip install -r requirements-pi.txt
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

## Study Mode

Study mode is a voice-activated camera monitor that watches head pose and flags repeated looking-away behavior. Start it by saying `aegis enable study mode` after the wake word, and stop it with `aegis disable study mode`.

When it starts, the assistant runs a short calibration pass. During calibration, sit normally and look at the screen so the system can learn your neutral yaw/pitch baseline. After calibration, the monitor samples frames at the configured FPS, compares each frame against that baseline, and emits a distraction event if you stay outside the safe zone long enough.

If repeated distraction events happen in the rolling window, the monitor emits an escalation event that is routed back through the existing orchestrator/LLM path.

### Camera backends

The camera path is selected with `camera_backend`:

- `auto` tries Picamera2 first when a CSI camera is detected, then falls back to a webcam.
- `webcam` forces OpenCV `VideoCapture` for laptop or USB-camera testing.
- `picamera` forces the Pi 5 CSI path and raises a descriptive error if Picamera2 or the camera is missing.

Auto-detection is intentionally conservative. On the Raspberry Pi 5, the CSI camera should be opened through Picamera2/libcamera instead of `cv2.VideoCapture`, because the latter is the wrong stack for the native Pi camera pipeline. If Picamera2 is not installed or reports no CSI camera, the code falls back to a webcam in `auto` mode and fails loudly only when you explicitly request `picamera`.

### Configuration

All of these values can be set through environment variables in the same style as the rest of the repo:

- `camera_backend` / `AEGIS_CAMERA_BACKEND`: `auto`, `webcam`, or `picamera` (`auto`)
- `camera_fps` / `AEGIS_CAMERA_FPS`: sampling rate for study mode frames (`10`)
- `calibration_duration_sec` / `AEGIS_CALIBRATION_DURATION_SEC`: neutral baseline capture time (`5.0`)
- `yaw_tolerance_deg` / `AEGIS_YAW_TOLERANCE_DEG`: yaw safe-zone tolerance (`12.0`)
- `pitch_tolerance_deg` / `AEGIS_PITCH_TOLERANCE_DEG`: pitch safe-zone tolerance (`10.0`)
- `distraction_duration_threshold_sec` / `AEGIS_DISTRACTION_DURATION_THRESHOLD_SEC`: sustained looking-away time before a distraction event (`3.0`)
- `escalation_event_count` / `AEGIS_ESCALATION_EVENT_COUNT`: distraction count needed to escalate (`3`)
- `escalation_window_sec` / `AEGIS_ESCALATION_WINDOW_SEC`: rolling window for escalation counting (`300.0`)
- `show_debug_window` / `AEGIS_SHOW_DEBUG_WINDOW`: enable the OpenCV debug window (`True`)

### Testing and deployment

- On a laptop, set `camera_backend=webcam` if you want to force the USB/laptop camera path.
- On the Raspberry Pi 5, keep `camera_backend=auto` or set `picamera` explicitly once the CSI camera is known to work.
- If the Pi camera is not detected, check the ribbon cable seating, confirm `libcamera-hello` works, and verify the camera shows up in Picamera2/libcamera tooling.
- If you are running headless on the Pi, set `show_debug_window=False` so OpenCV does not try to open a GUI window.

## Study mode config reference

The new study-mode values live in `config.py` alongside `AgentConfig` and `GoogleConfig`. Their defaults are the same as the environment-variable list above.

## Notes

- `config.py` loads `.env` automatically with `python-dotenv`.
- `.env`, `credentials/`, and Google token files are ignored by Git.
- `tools/query_rag.py` remains separate from the Google integrations.
- The old synthetic test harness under `tests/` was removed from this trimmed runtime snapshot.
