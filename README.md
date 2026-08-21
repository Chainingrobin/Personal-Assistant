# Personal Assistant

A local, LLM-centered personal assistant built for a Raspberry Pi 5 target. The runtime stays deliberately simple: Ollama for routing, plain Python tool functions for side effects, and direct Google API calls for email and calendar access.

The repo now also includes a separate vision path for study mode: a camera backend abstraction, MediaPipe-based head-pose estimation, and a monitor that can run while the assistant is otherwise idle.

## Setup

### 1. Install Ollama and pull the models

```bash
ollama pull qwen3:1.7b
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

## Text-to-Speech Setup

TTS is local model inference. The Python package requires the system
dependency `espeak-ng`, which is not pip-installable.

On Windows, install it with either:

```powershell
winget install --id eSpeak-NG.eSpeak-NG
```

or the Windows MSI from the [eSpeak NG releases](https://github.com/espeak-ng/espeak-ng/releases).
On Raspberry Pi OS, install it with:

```bash
sudo apt update
sudo apt install espeak-ng
```

Download the Kokoro assets into the configured locations:

```bash
python scripts/download_tts_models.py
```

The model files are machine-local and are ignored by Git. Configure TTS in
`.env` when the defaults do not match the machine:

```dotenv
TTS_BACKEND=kokoro
TTS_MODEL_PATH=models/tts/kokoro-v1.0.int8.onnx
TTS_VOICES_PATH=models/tts/voices-v1.0.bin
TTS_VOICE=af_sky
TTS_SPEED=1.0
```

The backend exposes `synthesize(text)` and returns raw audio samples with a
sample rate. Audio playback and orchestrator wiring are intentionally kept at
the application integration point rather than inside the backend.

### 4. Enroll each user once on a laptop

- Run the assistant on a machine with a browser.
- Trigger a Google-backed tool for that user once.
- Complete the consent flow.
- The app writes a per-user token file to `credentials/tokens/<user_id>.json`.
- Copy that token file to the Raspberry Pi for the same user.

The Pi should not need to open a browser for OAuth. In normal deployment, tokens already exist before the Pi runs the assistant.

### 5. Run the assistant

```bash
python main.py
```

Use `python orchestrate.py` only for the standalone LLM/tool-routing
diagnostic described below. The production runtime is `main.py`.

## How it is organized

```
Personal-Assistant/
├── config.py            # Root config for Ollama and Google settings.
├── credentials/         # Local-only OAuth material.
│  ├── credentials.json  # Google OAuth Desktop client secret.
│  └── tokens/           # Per-user token files: <user_id>.json.
├── main.py              # Production runtime and voice/text state machine.
├── orchestrate.py       # One-turn Ollama routing, tool dispatch, and summary.
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

## Architecture Overview

The assistant is a single-process Python application with clear boundaries
between input, control intents, model orchestration, integrations, and output:

```text
Voice or text input
        |
        v
Control intents (study mode / user switch)
        |
        v
Optional user-scoped context lookup (RAG)
        |
        v
Ollama tool selection ---> Google APIs or local RAG
        |                         |
        +------ concise response -+
                    |
             Kokoro TTS + playback
                    |
             OLED or emulator display
```

The application has two long-lived activity paths. The command path is
serialized by one heavy-task lock because Whisper, speaker verification,
Ollama, TTS, and playback are expensive and should not compete on the Pi.
Study mode runs camera capture and head-pose analysis in a daemon thread; only
its escalation handoff to the LLM uses that same lock.

### Crucial files and roles

- **`main.py` is the production entry point.** It initializes the configured
  components, chooses voice or text mode, owns the command state machine, and
  connects transcript handling to identity, RAG, orchestration, TTS, and the
  display. In voice mode the sequence is wake word -> VAD recording -> Whisper
  transcription -> direct control-intent handling -> normal command dispatch.
- **`orchestrate.py` is the model boundary.** `Orchestrator.run()` performs a
  blocking Ollama tool-selection pass, injects the active `user_id`, executes
  the selected Python tool, then performs a short streaming summarization
  pass. If no tool is selected, it returns the model's conversational reply;
  if the model returns neither a tool call nor text, it uses the streaming
  fallback. `AgentRequest` and `AgentResponse` are the stable request/response
  contract around this work.
- **`config.py` is the configuration boundary.** It loads `.env` once and
  exposes immutable `AgentConfig`, `GoogleConfig`, `VisionConfig`,
  `DisplayConfig`, and `TTSConfig` objects. The active user is not selected by
  the LLM; `main.py` passes it into the orchestrator and tools.
- **`audio/` is the voice input pipeline.** `wake_word.py` listens for the
  local ONNX wake-word model, `vad_recorder.py` finds utterance boundaries,
  `resample.py` normalizes microphone rates, and `stt.py` converts the final
  16 kHz mono audio to text with faster-whisper.
- **`identity/` is the speaker and profile layer.** `intent_parser.py`
  recognizes switch requests, while `voice_id.py` verifies the claimed user
  with ECAPA-TDNN against `identity/profiles/*.npy`. Voice mode verifies before
  switching; text mode intentionally switches directly because it has no
  speaker signal.
- **`rag/` and `tools/query_rag.py` provide local knowledge retrieval.**
  `ingest.py` creates embedded JSON records, `retriever.py` ranks them with
  cosine similarity, and the tool wrapper formats the results for the model.
  The normal command path skips RAG for date, time, email, and calendar
  keywords to avoid unnecessary embedding inference.
- **`tools/` contains side effects.** Gmail tools read mail or create drafts;
  Calendar tools read or create events; `query_rag.py` reads local knowledge.
  Email is never sent by the current draft tool. Tool exceptions are converted
  into an `[ERROR]` result so the model can report the failure.
- **`vision/` is the independent study-mode monitor.** Camera backends provide
  frames, `head_pose.py` estimates yaw/pitch, and `study_mode.py` calibrates a
  neutral baseline, detects sustained looking away, and escalates repeated
  distractions within a rolling window.
- **`tts/` and `display/` are output adapters.** Kokoro synthesizes local audio
  and `playback.py` plays it. The display factory selects the SSD1306 hardware
  renderer or a laptop emulator; `display/states.py` is the central mapping of
  runtime states to icons and subtitles.

### Runtime modes

#### Voice mode: `INPUT_MODE=voice`

This is the default deployment mode. The wake-word listener remains active,
then the recorder captures one utterance. Empty audio and empty transcripts
return to `IDLE`. A recognized study-mode command or user-switch command is
handled locally; all other transcripts go through RAG, Ollama, tools, and TTS.

#### Text mode: `INPUT_MODE=text`

`TextInputSource` reads one command at a time from stdin. This is useful for
laptop, headless, and dependency-isolation testing because it bypasses the
microphone, wake word, VAD, Whisper, and voice verification. It still supports
study-mode control, user switching, RAG, Google tools, TTS, and display updates.
EOF or Ctrl+C exits the loop and stops study mode.

#### Study mode

Saying `aegis enable study mode` starts camera calibration and then background
monitoring. Saying `aegis disable study mode` stops it. A distraction event is
logged and produces a short notification tone. Once the configured rolling
count is reached, an escalation displays `Stay focused!` and sends a coaching
request to Ollama while the monitor remains active. The current escalation
handler prints the model response through the orchestrator but does not send it
through TTS; normal user commands do use TTS.

### Possible outputs

Depending on the path taken, the system can produce:

- A direct date/time answer from the dynamically built system context, without
  calling a tool.
- A concise natural-language answer after a Gmail, Calendar, or RAG tool call.
- A local Gmail draft; the system does not send email.
- A successful or failed user-switch result based on voice similarity.
- Study-mode status, distraction notifications, or an LLM-generated coaching
  message.
- Terminal logs containing state transitions, timing measurements, tool output,
  errors, and streamed model text.
- Spoken output through Kokoro and visual status output through the OLED or
  emulator. Display states include `IDLE`, `LISTENING`, `THINKING`, tool states,
  calibration, study mode, user-switch success/failure, distraction alert, and
  error.

### Data and security boundaries

- `credentials/credentials.json` is the Google OAuth client secret and
  `credentials/tokens/<user_id>.json` stores per-user refresh tokens. These are
  local-only and ignored by Git.
- `userclips/` contains raw biometric recordings and
  `identity/profiles/*.npy` contains speaker embeddings. Both are ignored by
  Git; profiles should still be treated as sensitive personal data.
- `models/` contains machine-local model assets for TTS and wake-word
  detection. RAG JSON records under `rag/db/` are tracked and may contain
  user-specific facts, so access to the repository should be controlled.
- The active `user_id` is supplied by the runtime after identity handling. It
  is passed to Google tools, but the current `rag/retriever.py` loads and
  ranks every JSON record without filtering by user. This is the most important
  privacy-related cleanup item before broader deployment.

## Runtime model

- `user_id` is injected by the orchestrator after the model picks a tool.
- The model handles tool selection and task arguments only.
- User identity comes from the face/voice layer elsewhere in the system.
- Google OAuth tokens are stored per user so one person’s account never overwrites another’s.
- The laptop is used for first-time consent; the Pi only reuses copied token files.

## Display

This project targets a 0.96" SSD1306 I2C monochrome OLED panel at 128x64 with a 4-wire connection: VCC, GND, SDA, and SCL. The display module is intentionally backend-agnostic so the same code runs with either a real adapter on the Raspberry Pi or a local emulator window on a laptop.

Use the environment switch to choose the rendering path:

```bash
AEGIS_DISPLAY_BACKEND=emulator
# or on the Pi:
AEGIS_DISPLAY_BACKEND=ssd1306
```

The `.env` file can set either value. The module uses `AEGIS_DISPLAY_I2C_PORT` and `AEGIS_DISPLAY_I2C_ADDR` to locate the device on the Pi; the default address is `0x3C`.

To verify wiring on the Pi once the hardware arrives:

```bash
sudo raspi-config
# enable I2C

i2cdetect -y 1
```

If the SSD1306 is connected correctly, it should appear at `0x3C`.

To run the emulator locally:

```bash
AEGIS_DISPLAY_BACKEND=emulator python main.py
```

This opens a pygame window showing the same 128x64 render path the Pi uses. To add a new display state, add the bitmap to `display/icons.py`, add the enum member and `STATE_CONFIG` entry in `display/states.py`, and the renderer consumes it automatically without any other call-site changes.

## Study Mode

Study mode is a voice-activated camera monitor that watches head pose and flags repeated looking-away behavior. Start it by saying `aegis enable study mode` after the wake word, and stop it with `aegis disable study mode`.

When it starts, the assistant runs a short calibration pass. During calibration, sit normally and look at the screen so the system can learn your neutral yaw/pitch baseline. After calibration, the monitor samples frames at the configured FPS, compares each frame against that baseline, and emits a distraction event if you stay outside the safe zone long enough.

If repeated distraction events happen in the rolling window, the monitor emits an escalation event that is routed back through the existing orchestrator/LLM path.

## Camera Setup

The camera path is selected with `AEGIS_CAMERA_BACKEND`:

- `auto` checks for `rpicam-vid` and a detected CSI camera first, then falls back to a webcam.
- `webcam` forces OpenCV `VideoCapture` for laptop or USB-camera testing.
- `picamera` forces the Pi 5 CSI path and raises a descriptive error if `rpicam-vid` or the camera is missing.

On this repo's pyenv-managed Python 3.11.9 venv, the Raspberry Pi camera path does not import Picamera2/libcamera Python bindings at all. Those bindings are compiled against the system Python ABI on Debian, so `import libcamera` fails inside the venv even though the camera hardware is present. The repo now uses `rpicam-vid` as a subprocess and pipes raw YUV420/I420 frames over stdout instead. That keeps the existing venv intact and avoids the ABI mismatch entirely.

If the Pi camera stops opening, run the repo-local diagnostic script once from the activated venv:

```bash
python scripts/diagnose_camera.py
```

The script prints Python version, `cv2`, `picamera2` import behavior, `rpicam-vid` discovery, camera enumeration, `/dev/video*`, and group membership in one pass.

If the OpenCV window cannot open because Qt/Wayland is unavailable, study mode writes the latest annotated preview frame to `.aegis_debug/study_mode_latest.jpg` and logs that path periodically while calibration is running.

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

`requirements.txt` deliberately pins `mediapipe==0.10.9`. Avoid upgrading it casually when adding packages, because protobuf resolver changes have silently bumped it in the past and broken the vision stack.

### Testing and deployment

- On a laptop, set `camera_backend=webcam` if you want to force the USB/laptop camera path.
- On the Raspberry Pi 5, keep `camera_backend=auto` or set `picamera` explicitly once the CSI camera is known to work.
- If the Pi camera is not detected, check the ribbon cable seating, confirm `rpicam-vid --list-cameras` works, and rerun `python scripts/diagnose_camera.py`.
- If you are running headless on the Pi, set `show_debug_window=False` so OpenCV does not try to open a GUI window.

## Study mode config reference

The new study-mode values live in `config.py` alongside `AgentConfig` and `GoogleConfig`. Their defaults are the same as the environment-variable list above.

## Notes

- `config.py` loads `.env` automatically with `python-dotenv`.
- `.env`, `credentials/`, and Google token files are ignored by Git.
- `tools/query_rag.py` remains separate from the Google integrations.
- The old synthetic test harness under `tests/` was removed from this trimmed runtime snapshot.

## Voice ID (speaker verification)

Voice ID handles the "switch user X" flow: it verifies that the person
claiming a profile switch is actually who they say they are, using a
speaker-verification model (ECAPA-TDNN) rather than trusting the spoken
name alone. It only runs when a switch is requested — it is not a
continuously-running background process.

### Files

- **`identity/voice_id.py`** — Core module. Loads the ECAPA-TDNN encoder
  once (`SpeakerVerifier`), embeds audio into a fixed-length vector, and
  compares a live utterance against a saved profile via cosine similarity.
  Also owns the "currently active user" state (`get_current_user()` /
  `set_current_user()`) and the public `switch_user()` function that
  orchestrate.py-adjacent code calls to attempt a switch. If the claimed
  user is already active, `switch_user()` short-circuits and skips
  re-verifying (pass `force_verify=True` to override this during testing).

- **`identity/enroll_record.py`** — One-time enrollment CLI. Takes a user_id and a
  folder of WAV clips, embeds each clip, averages them into one profile
  vector, and saves it to `identity/profiles/<user_id>.npy`. Re-running it
  for the same user overwrites their old profile — there is no separate
  "unenroll" step needed.

- **`identity/diagnose.py`** — Combined enroll + consistency check. Does
  everything `enroll.py` does, but also runs a leave-one-out test across
  your clips: it holds out each clip in turn, builds a profile from the
  rest, and scores the held-out clip against it. This flags any single
  clip that looks like an outlier (bad take, background noise) before you
  trust the profile.

- **`identity/live_test_verify.py`** — Live-mic version of the above. Uses
  the repo's real `VADRecorder` (same silence-based end-of-utterance
  detection the wake-word pipeline uses in production) to capture each
  utterance, then runs it through the real `switch_user()` call — so this
  is the closest thing to testing the actual production path without
  running the full wake-word/STT chain. Supports `--force` to bypass the
  already-active-user short-circuit for testing purposes, and `--vad-debug`
  for frame-level VAD logging.

- **`vad_recorder.py`** — Shared microphone/VAD utility used by both the
  live test script and the production wake-word pipeline. Records audio
  and returns once silence has lasted long enough to mark end-of-utterance.
  Not voice-ID-specific, but voice ID depends on it for live capture.

### Setup

```bash
pip install speechbrain soundfile sounddevice webrtcvad torch
```

The first run of any script that loads `SpeakerVerifier` downloads the
ECAPA-TDNN weights (~80MB) into the local Hugging Face cache
(`~/.cache/huggingface`). This requires network access once; after that
it's fully local.

> **Windows + conda users:** if model loading fails with an SSL/
> `SSL_CERT_FILE` error, this is caused by conda's `base` environment
> auto-activating alongside your `.venv` and injecting a stale cert path.
> Run `conda config --set auto_activate_base false` once (restart your
> terminal after), or run `conda deactivate` before activating `.venv`
> each session.

### Enrollment workflow (run once per teammate)

1. **Record clips.** 5–8 WAV files, 4–6 seconds each, per person. Mix a
   few actual switch-phrases ("switch user \<name\>") with a few unrelated
   natural sentences — this keeps the profile from overfitting to just
   the trigger phrase. Record using the **same mic you'll test/deploy
   with** — mic mismatch between enrollment and verification is the
   single biggest source of false rejections we've seen in testing
   (recording enrollment on a phone and testing live off a laptop mic
   dropped scores by ~0.15–0.2). If you plan to test across multiple mics
   (e.g. laptop mic and a headset/IEM mic), record a few clips on each
   and include them all in the same enrollment folder — this gives the
   averaged profile more robustness to condition changes instead of
   overfitting to one mic's frequency response.

   Save clips as WAV in `userclips/<your_name>/`. If recording on a phone
   (e.g. iOS Voice Memos exports `.m4a`), convert first:

   ```bash
   ffmpeg -i clip1.m4a -ar 16000 -ac 1 clip1.wav
   ```

2. **Enroll + diagnose:**

   ```bash
   python identity/diagnose.py <your_name> userclips/<your_name>
   ```

   This saves `identity/profiles/<your_name>.npy` and prints a
   leave-one-out score per clip. If one clip scores noticeably lower than
   the rest, re-record or drop it and re-run — safe to run as many times
   as needed, it just overwrites.

3. **Live verification test:**

   ```bash
   python identity/live_test_verify.py <your_name>
   ```

   Press Enter, speak (ideally "switch user \<your_name\>"), and it'll
   auto-stop on silence and print a similarity score. Repeat several
   times across a normal speaking session to get a feel for your typical
   score range before trusting the threshold.

   To test whether the system correctly _rejects_ a mismatched voice
   (disguised voice, or someone else's clip), add `--force` so it always
   runs a real comparison instead of short-circuiting when the claimed
   user is already active:

   ```bash
   python identity/live_test_verify.py <your_name> --force
   python identity/live_test_verify.py youssef --force
   ```

4. **Full pipeline test.** Once steps 2–3 look solid, run the full
   assistant (`main.py`) and say the wake word followed by
   "switch user \<your_name\>" live, to confirm the whole
   wake-word → VAD → Whisper → ECAPA chain works end-to-end, not just the
   isolated identity layer.

### Threshold

Default is `0.70` (cosine similarity), overridable via
`AEGIS_VOICE_ID_THRESHOLD`. Don't tune this in isolation — if genuine-voice
scores are landing inconsistently near the threshold, first check for a
mic/condition mismatch between enrollment and testing (see step 1) rather
than lowering the threshold to compensate. A looser threshold makes
imposter rejection weaker system-wide, so it should be a deliberate
calibration decision based on stable, matched-condition scores — not a
band-aid for an enrollment recorded under different conditions than it's
being tested against.

### .gitignore

- **`userclips/`** — ignore entirely. Raw voice recordings are personal
  biometric data and shouldn't sit in git history.
- **`identity/profiles/*.npy`** — these are compressed embeddings, not raw
  audio, so the privacy risk is much lower (you can't reconstruct a
  voice from one). Whether to commit them is a team call — committing
  them lets everyone test switching without each person re-enrolling
  locally, but check with each teammate before pushing their profile.

### Project description

Personal Assistant, also referred to in the runtime as Aegis, is a privacy-
oriented, local-first personal assistant designed for a Raspberry Pi 5. It
combines natural-language interaction, personal productivity integrations,
retrieval-augmented context, speaker verification, and a camera-based study
mode. The central design goal is to keep computation and personal data local
where practical while using direct, narrowly scoped integrations for Gmail and
Google Calendar.

The assistant is not a single chatbot prompt. It is an event-driven system
with specialized components: audio input turns speech into a transcript,
identity logic determines the active user, RAG supplies relevant personal
context, Ollama decides whether a tool is needed, tools perform controlled
actions, and local TTS/display adapters return the result to the user.

### Problem being addressed

Users often need to switch between calendars, emails, personal notes, tasks,
and study support. Conventional assistants may depend heavily on cloud
services, provide limited user separation, or lack a focused workflow for
study sessions. This project addresses those needs with one assistant that can:

- Answer natural-language questions about personal information.
- Read Gmail and Calendar data through explicit tools.
- Create Calendar events and Gmail drafts without automatically sending email.
- Retrieve relevant local notes and preferences using embeddings.
- Support multiple users with per-user Google OAuth tokens and voice profiles.
- Detect repeated distraction during study sessions using head pose.
- Run in a voice-controlled Pi deployment or a text-driven laptop/headless mode.

### Architecture to present

Present the system as six cooperating layers:

1. **Input layer:** `audio/wake_word.py`, `audio/vad_recorder.py`,
   `audio/stt.py`, and `text_input.py` accept either voice or text.
2. **Identity and control layer:** `identity/intent_parser.py` handles explicit
   mode/user commands, and `identity/voice_id.py` verifies a claimed speaker
   before a voice-mode user switch.
3. **Context layer:** `rag/ingest.py` creates embedded JSON records and
   `rag/retriever.py` ranks relevant records using cosine similarity.
4. **Reasoning and action layer:** `orchestrate.py` is the Ollama boundary. It
   selects a tool, injects the active user, executes the tool, and summarizes
   the result.
5. **Integration layer:** `tools/` contains isolated Gmail, Calendar, and RAG
   functions; `tools/auth.py` manages Google OAuth credentials.
6. **Output and monitoring layer:** `tts/` speaks normal responses,
   `display/` renders runtime state, and `vision/` runs study-mode camera
   monitoring independently of ordinary command processing.

### Main execution story

The production entry point is `main.py`. At startup it loads configuration,
creates the Ollama transport and orchestrator, initializes TTS and display,
and then selects the configured input mode.

In voice mode, the flow is:

```text
Wake word -> microphone recording with VAD -> Whisper transcript
          -> study/user control check -> optional RAG retrieval
          -> Ollama tool decision -> tool execution -> response summary
          -> Kokoro speech + OLED/emulator status -> idle
```

In text mode, stdin supplies the transcript and the same control and dispatch
logic is reused after the audio stages are bypassed. This makes text mode a
practical development and headless deployment path rather than a separate
assistant implementation.

### Why `orchestrate.py` matters

`orchestrate.py` is the reusable one-turn intelligence layer, not the full
application loop. Its key components are:

- `AgentRequest`: messages and available callable tools.
- `OllamaTransport`: blocking tool-selection calls and streaming summary calls.
- `TOOL_REGISTRY`: the approved mapping from model-selected names to Python
  functions.
- `Orchestrator.run()`: retries the model pass, dispatches at most the first
  returned tool call, attaches `user_id`, catches tool errors, and returns an
  `AgentResponse`.
- `AgentResponse`: response text plus tool name, arguments, raw result,
  reasoning text, attempt count, and success status.

The two-pass design is worth highlighting in a presentation. The first call
is optimized for structured tool selection. When a tool is selected, the
second call receives only the tool name and result and streams a concise
human-facing summary. This reduces unnecessary context processing on the Pi.
When no tool is required, the first response is returned directly; an empty
response triggers a streaming fallback.

### Demonstration scenarios

Use these scenarios to demonstrate the system in a logical order:

1. **Direct answer:** Ask for today's date or tomorrow's date. The system
   answers from the dynamic prompt context and deliberately avoids a calendar
   API call.
2. **Personal retrieval:** Ask about a stored preference or responsibility.
   The command path queries RAG, injects the retrieved context into the system
   prompt, and produces a concise answer.
3. **Calendar lookup:** Ask what events are scheduled today. Ollama selects
   `get_calendar_events`, the tool calls Google Calendar, and the model
   summarizes the result.
4. **Safe email action:** Ask to draft an email. The Gmail tool creates a draft
   only, demonstrating that the action boundary is narrower than unrestricted
   email sending.
5. **User switching:** Say a switch-user command. The parser identifies the
   target and ECAPA-TDNN compares the utterance against the target profile
   before changing the active user.
6. **Study mode:** Enable study mode, complete camera calibration, look away
   long enough to trigger a distraction, and show the display/notification
   behavior. Repeated events eventually generate a coaching escalation.
7. **Text/headless mode:** Set `INPUT_MODE=text` and run the same calendar or
   RAG scenario without microphone hardware.

### Presentation talking points

- **Local-first design:** Ollama, embeddings, Whisper, speaker verification,
  Kokoro, and most control logic run locally.
- **Separation of responsibilities:** `main.py` controls lifecycle and input;
  `orchestrate.py` controls one model turn; tools own side effects; adapters
  isolate hardware and output choices.
- **Explicit action control:** The model can select only registered tools, and
  the runtime supplies the active user rather than allowing the model to choose
  an account identity.
- **Resource awareness:** Pi-specific model settings, lazy/one-time component
  initialization, short prompts, streaming summaries, native audio-rate
  capture, and a shared heavy-task lock reduce contention.
- **Multimodal interaction:** Voice, text, audio output, OLED status, and camera
  monitoring coexist in one runtime.
- **Graceful operating modes:** Hardware-dependent voice/camera features can be
  bypassed for development with text input, webcam selection, debug previews,
  and the display emulator.
- **Privacy boundary:** OAuth tokens and raw voice recordings are local and
  ignored by Git, while tool permissions and draft-only email reduce accidental
  side effects.

### Technical tradeoffs to explain

- A local model improves privacy and offline capability but has less capacity
  than a large hosted model and may occasionally choose the wrong tool.
- The two-pass tool flow improves response quality and readability but adds a
  second inference call after every tool invocation.
- Voice verification improves user separation but requires enrollment,
  microphone consistency, and local model downloads.
- Camera study mode is intentionally heuristic: it uses a calibrated neutral
  yaw/pitch baseline, so lighting, face visibility, posture, and camera angle
  affect reliability.
- Direct Google APIs provide predictable actions but require OAuth setup and
  cached per-user tokens.
- A 128x64 monochrome display is reliable and low power but communicates only
  compact status, not full response content.

### Current limitations and risks

These should be stated honestly in a report rather than presented as completed
features:

- The RAG wrapper accepts `user_id`, but the current retriever ranks all JSON
  records without applying a user filter. User-specific retrieval isolation is
  the highest-priority privacy improvement.
- The repository contains synthetic fixtures under `tests/` but no current
  Python unit or integration test suite.
- Study-mode escalation is sent to the orchestrator and printed/displayed, but
  its returned coaching text is not currently routed through TTS.
- Some configuration fields for Ollama, including base URL and timeout, are
  defined but are not consumed by the direct Python client call.
- Older documents still refer to removed `services/` packages and synthetic
  harness details, so documentation should be reconciled before submission.
- Camera, microphone, Google OAuth, Ollama, TTS model assets, and wake-word
  assets require environment-specific setup; a full hardware demo is not
  reproducible from Python dependencies alone.

### Recommended future work

Prioritize improvements in this order:

1. Enforce `user_id` filtering in RAG and add tests proving cross-user data is
   not returned.
2. Add unit tests for intent parsing, timeframe parsing, cosine ranking,
   camera selection, and display rendering.
3. Remove duplicated tool lists and obsolete diagnostic code.
4. Wire configured Ollama connection settings into the transport or remove
   unused settings.
5. Decide whether study coaching should be spoken and connect it to TTS if it
   should be audible.
6. Update historical documentation and add a deployment checklist for the Pi.
7. Measure latency, CPU, memory, and false acceptance/rejection rates under
   realistic microphone and camera conditions.

### Suggested presentation structure

For a short technical presentation, use this sequence:

1. Motivation and project goals.
2. User interaction and representative use cases.
3. High-level architecture diagram.
4. `main.py` runtime flow and the voice/text modes.
5. `orchestrate.py`, tool calling, RAG, and Google integrations.
6. Voice identity and privacy boundaries.
7. Study mode and multimodal output.
8. Raspberry Pi optimization decisions.
9. Live demonstration or recorded workflow.
10. Limitations, testing status, and future work.

The central conclusion should be: this repository demonstrates a modular,
local-first assistant architecture in which an LLM handles language and tool
selection, while deterministic Python components retain control over identity,
side effects, hardware, and user-visible state.
