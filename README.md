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

- **`identity/enroll.py`** — One-time enrollment CLI. Takes a user_id and a
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

- **`identity/test_verify.py`** — Scores one pre-recorded WAV file against
  a saved profile. Useful for quick, repeatable checks against a fixed
  clip, but does not reflect live-mic conditions.

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
