"""
Entry point / runtime loop for the Aegis assistant.

This file owns the full audio state machine:
wake word -> record utterance -> transcribe -> optional voice verification ->
orchestrate one command -> idle. The LLM orchestration itself still lives in
orchestrate.py and remains unchanged (aside from display-state calls).
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import AGENT_CONFIG, VISION_CONFIG, DISPLAY_CONFIG
from orchestrate import AgentRequest, OllamaTransport, Orchestrator

from audio.stt import WhisperTranscriber
from audio.vad_recorder import VADRecorder
from audio.wake_word import WakeWordListener
from identity.intent_parser import extract_switch_user_intent
from identity.intent_parser import extract_study_mode_intent
from identity.voice_id import (
    SpeakerVerifier,
    get_current_user,
    get_known_user_ids,
    set_current_user,
    switch_user,
)
from text_input import TextInputSource
from tts import TTSBackend, get_tts_backend, play
from tts.base import sanitize_for_speech
from vision.study_mode import StudyModeEvent, StudyModeMonitor
from tools.add_calendar_event import add_calendar_event
from tools.auth import get_google_credentials
from tools.draft_email import draft_email
from tools.get_calendar_events import get_calendar_events
from tools.query_rag import query_rag
from tools.read_email import read_email

from display import init_display, set_state, shutdown as display_shutdown
from display.states import DisplayState

init_display(DISPLAY_CONFIG)


ALL_TOOLS = [read_email, get_calendar_events, add_calendar_event, draft_email]


def ensure_user_enrolled(user_id: str) -> None:
    """Check Google auth for one user_id and refresh or enroll if needed.

    Input: user_id string such as 'robin'. Output: none; raises on auth errors.
    On the Pi this should normally be a cache-hit and take well under 1 second.
    """
    print(f"Checking Google auth for {user_id}...", end=" ", flush=True)
    try:
        get_google_credentials(user_id)
        print("OK")
    except Exception as exc:
        print(f"FAILED ({exc})")
        raise


def build_system_prompt(user_id: str, rag_context: str = "") -> str:
    """Build the LLM system prompt for one command turn.

    Input: current user_id and optional RAG text. Output: a single prompt string.
    This is CPU-light string assembly only; no model inference happens here.
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day_of_week = now.strftime("%A")
    tomorrow_day = (now + timedelta(days=1)).strftime("%A")
    current_time = now.strftime("%I:%M %p")

    rag_section = ""
    if rag_context and rag_context.strip():
        rag_section = f"\nUSER CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}\n"

    return f"""You are Aegis, an intelligent personal assistant for {user_id}.

CURRENT CONTEXT:
- Today's Date: {today_str} ({day_of_week})
- Tomorrow's Date: {tomorrow_str} ({tomorrow_day})
- Current Time: {current_time}
{rag_section}
TOOL ROUTING RULES (STRICT, IN PRIORITY ORDER):
1. HIGHEST PRIORITY: if the user is only asking what today's or tomorrow's
   date/day/time is (nothing about events, plans, or schedule), answer
   DIRECTLY from CURRENT CONTEXT above. Never call a tool for this, even
   though get_calendar_events also deals with dates.
2. If the user asks about scheduled events, appointments, or their calendar:
   call get_calendar_events.
3. If the user asks about messages or emails: call read_email.

EXAMPLES:
User: "what's today's date"
Assistant: (no tool call) "Today is {today_str}."

User: "what's tomorrow's date"
Assistant: (no tool call) "Tomorrow is {tomorrow_str} ({tomorrow_day})."

User: "what day is tomorrow"
Assistant: (no tool call) "Tomorrow is {tomorrow_day}."

User: "do I have anything today"
Assistant: (calls get_calendar_events with start_date='{today_str}')

User: "what events do I have in all of September 2026"
Assistant: (calls get_calendar_events with start_date='2026-09-01', end_date='2026-09-30')

Never say "I don't have access to your calendar or email." You have direct access via tools.
"""


def run_one_turn(orchestrator: Orchestrator, user_id: str, user_input: str):
    """Run one orchestrated assistant turn for one user message.

    Input: orchestrator instance, active user_id, and one transcript string.
    Output: AgentResponse from the tool-routing layer.
    This is the only place the heavy LLM runs; Whisper/ECAPA stay outside.
    """
    skip_rag_keywords = {"date", "time", "today", "tomorrow", "email", "emails", "calendar", "events", "schedule"}
    words = set(user_input.lower().split())
    should_query_rag = not words.intersection(skip_rag_keywords)

    rag_context = ""
    if should_query_rag:
        rag_context = query_rag(query=user_input, top_k=3, user_id=user_id)

    request = AgentRequest(
        messages=[
            {"role": "system", "content": build_system_prompt(user_id, rag_context=rag_context)},
            {"role": "user", "content": user_input},
        ],
        tools=[read_email, get_calendar_events, add_calendar_event, draft_email],
    )
    return orchestrator.run(request, current_user_id=user_id)


@dataclass
class RuntimeComponents:
    """Bundle the heavy runtime components that should be initialized once.

    The wake-word listener stays always on; the recorder, Whisper model, and
    ECAPA encoder are invoked sequentially after detection, never concurrently.
    """

    orchestrator: Orchestrator
    wake_listener: WakeWordListener
    recorder: VADRecorder
    transcriber: WhisperTranscriber
    voice_verifier: SpeakerVerifier
    study_mode_monitor: StudyModeMonitor
    tts_backend: TTSBackend
    heavy_task_lock: threading.Lock



def _handle_wake_event(runtime: RuntimeComponents) -> None:
    t0 = time.perf_counter()
    set_state(DisplayState.LISTENING)
    print("[state] RECORDING — capturing utterance...")
    with runtime.heavy_task_lock:
        utterance = runtime.recorder.record_utterance()
    t1 = time.perf_counter()
    print(f"[timing] VAD record: {t1 - t0:.2f}s")

    if utterance is None or utterance.samples.size == 0:
        print("[audio] No speech captured; returning to idle.")
        set_state(DisplayState.IDLE)
        return

    print("[state] TRANSCRIBING...")
    with runtime.heavy_task_lock:
        transcript = runtime.transcriber.transcribe(utterance.samples, sample_rate=utterance.sample_rate).strip()
    t2 = time.perf_counter()
    print(f"[stt] {transcript}")
    print(f"[timing] Whisper STT: {t2 - t1:.2f}s")

    if not transcript:
        print("[stt] Empty transcript; returning to idle.")
        set_state(DisplayState.IDLE)
        return

    study_mode_intent = extract_study_mode_intent(transcript)
    if study_mode_intent == "enable":
        print("[study] Enabling study mode monitor...")
        try:
            runtime.study_mode_monitor.start()
            
        except Exception as exc:
            print(f"[study] Failed to enable study mode: {exc}")
            set_state(DisplayState.IDLE)
        return
    if study_mode_intent == "disable":
        print("[study] Disabling study mode monitor...")
        try:
            runtime.study_mode_monitor.stop()
        except Exception as exc:
            print(f"[study] Failed to disable study mode cleanly: {exc}")
        finally:
            set_state(DisplayState.IDLE)
        return

    claimed_user = extract_switch_user_intent(transcript, known_user_ids=get_known_user_ids())
    if claimed_user:
        print(f"[state] VERIFYING — switch intent for '{claimed_user}'...")
        set_state(DisplayState.SWITCHING_USER)
        t_verify0 = time.perf_counter()
        with runtime.heavy_task_lock:
            success, score = switch_user(
                claimed_user, utterance.samples, sample_rate=utterance.sample_rate, verifier=runtime.voice_verifier,
            )
        t_verify1 = time.perf_counter()
        print(f"[timing] ECAPA verify: {t_verify1 - t_verify0:.2f}s")
        if success:
            ensure_user_enrolled(claimed_user)
            print(f"[identity] Switched to {claimed_user} with similarity={score:.3f}")
            set_state(DisplayState.SWITCH_SUCCESS)
        else:
            print(f"[identity] Voice verification failed for {claimed_user}; score={score:.3f}")
            set_state(DisplayState.SWITCH_FAILED) 
        time.sleep(1.0)
        set_state(DisplayState.IDLE)
        return

    active_user = get_current_user()
    print(f"[state] DISPATCHING — active user '{active_user}'...")
    t3 = time.perf_counter()
    with runtime.heavy_task_lock:
        # NOTE: THINKING / TOOL_* display states are set inside Orchestrator.run
        # itself (orchestrate.py), since that's where tool-call detection happens.
        response = run_one_turn(runtime.orchestrator, active_user, transcript)
        audio = runtime.tts_backend.synthesize(response.raw_output)
        play(audio)
    t4 = time.perf_counter()
    print(f"[timing] Orchestrator (RAG+LLM+tools): {t4 - t3:.2f}s")
    print(f"[timing] TOTAL turn: {t4 - t0:.2f}s")
    set_state(DisplayState.IDLE)


def _handle_text_transcript(
    orchestrator: Orchestrator,
    study_mode_monitor: StudyModeMonitor,
    tts_backend: TTSBackend,
    heavy_task_lock: threading.Lock,
    transcript: str,
) -> None:
    """Handle one text transcript using the same control/dispatch rules as voice."""
    if not transcript:
        return

    study_mode_intent = extract_study_mode_intent(transcript)
    if study_mode_intent == "enable":
        print("[study] Enabling study mode monitor...")
        try:
            study_mode_monitor.start()
        except Exception as exc:
            print(f"[study] Failed to enable study mode: {exc}")
        return
    if study_mode_intent == "disable":
        print("[study] Disabling study mode monitor...")
        try:
            study_mode_monitor.stop()
        except Exception as exc:
            print(f"[study] Failed to disable study mode cleanly: {exc}")
        return

    claimed_user = extract_switch_user_intent(transcript, known_user_ids=get_known_user_ids())
    if claimed_user:
        # Text mode has no speaker signal to verify; voice mode remains unchanged above.
        set_current_user(claimed_user)
        ensure_user_enrolled(claimed_user)
        print(f"[identity] Switched to {claimed_user} from text input.")
        set_state(DisplayState.SWITCH_SUCCESS)   # NEW
        time.sleep(1.0)                            # NEW
        set_state(DisplayState.IDLE)   
        return

    active_user = get_current_user()
    print(f"[state] DISPATCHING — active user '{active_user}'...")
    with heavy_task_lock:
        response = run_one_turn(orchestrator, active_user, transcript)
        audio = tts_backend.synthesize(sanitize_for_speech(response.raw_output))
        play(audio)


def _handle_study_mode_event(event: StudyModeEvent, orchestrator: Orchestrator, heavy_task_lock: threading.Lock,tts_backend: TTSBackend,) -> None:
    if event.kind != "escalation":
        return

    set_state(DisplayState.DISTRACTION_ALERT)
    # Hold the alert on screen briefly — otherwise Orchestrator.run immediately
    # sets THINKING at the start of its reasoning pass and the alert never
    # actually becomes visible to the user.
    time.sleep(1.5)

    current_user = get_current_user()
    
    request = AgentRequest(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Aegis. The study-mode monitor detected repeated distractions. "
                    "Reply with a concise, task-priority-aware coaching message. "
                    "Do not mention internal thresholds or implementation details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User: {current_user}\n"
                    f"Event: repeated distraction in study mode\n"
                    f"Timestamp: {event.timestamp.isoformat()}\n"
                    f"Yaw: {event.yaw_deg:.2f}\n"
                    f"Pitch: {event.pitch_deg:.2f}\n"
                    f"Away duration: {event.away_duration_sec:.2f}s\n"
                    f"Rolling count: {event.rolling_window_count}"
                ),
            },
        ],
        tools=[],
    )

    print(f"[study] Routing escalation event to LLM for user '{current_user}'...")
    with heavy_task_lock:
        response = orchestrator.run(request, current_user_id=current_user)
        if response.raw_output.strip():
            audio = tts_backend.synthesize(sanitize_for_speech(response.raw_output))
            play(audio)

    # Study mode is still active after handling one distraction escalation,
    # so return the display to the STUDY_MODE indicator rather than IDLE.
    set_state(DisplayState.STUDY_MODE)


def _create_study_mode_monitor(orchestrator: Orchestrator, heavy_task_lock: threading.Lock,tts_backend: TTSBackend,) -> StudyModeMonitor:
    return StudyModeMonitor(
        config=VISION_CONFIG,
        on_event=lambda event: _handle_study_mode_event(event, orchestrator, heavy_task_lock, tts_backend),
        heavy_task_lock=heavy_task_lock,
    )


def main() -> None:
    """Boot the assistant and run the wake-word loop until interrupted.

    Output: none. This is the top-level Pi runtime entrypoint.
    """
    print(f"[config] model={AGENT_CONFIG.model} profile={AGENT_CONFIG.hardware_profile} temp={AGENT_CONFIG.temperature}")
    transport = OllamaTransport(model=AGENT_CONFIG.model)
    orchestrator = Orchestrator(transport, max_retries=5, verbose=True)
    tts_backend = get_tts_backend()
    heavy_task_lock = threading.Lock()

    if AGENT_CONFIG.input_mode == "text":
        # Local/dev/headless mode: bypass wake word, VAD, Whisper, and ECAPA entirely.
        study_mode_monitor = _create_study_mode_monitor(orchestrator, heavy_task_lock, tts_backend)
        ensure_user_enrolled(get_current_user())
        text_input = TextInputSource()
        print("Aegis text mode is running. Type a command; press Ctrl+C or send EOF to exit.")
        try:
            while True:
                transcript = text_input.read_transcript()
                if transcript is None:
                    break
                _handle_text_transcript(orchestrator, study_mode_monitor, tts_backend, heavy_task_lock, transcript)
        except KeyboardInterrupt:
            print("\n[main] Exiting on user interrupt.")
        finally:
            study_mode_monitor.stop()
        return

    voice_verifier = SpeakerVerifier()
    transcriber = WhisperTranscriber()
    recorder = VADRecorder()
    wake_listener = WakeWordListener()
    study_mode_monitor = _create_study_mode_monitor(orchestrator, heavy_task_lock, tts_backend)

    ensure_user_enrolled(get_current_user())
    set_state(DisplayState.IDLE)

    runtime = RuntimeComponents(
        orchestrator=orchestrator,
        wake_listener=wake_listener,
        recorder=recorder,
        transcriber=transcriber,
        voice_verifier=voice_verifier,
        study_mode_monitor=study_mode_monitor,
        tts_backend=tts_backend,
        heavy_task_lock=heavy_task_lock,
    )

    print("Aegis is running. Say 'aegis' to wake it up. Press Ctrl+C to exit.")

    def _on_wake() -> None:
        _handle_wake_event(runtime)

    try:
        runtime.wake_listener.listen(_on_wake)
    except KeyboardInterrupt:
        print("\n[main] Exiting on user interrupt.")
    finally:
        runtime.study_mode_monitor.stop()
        display_shutdown()


if __name__ == "__main__":
    main()