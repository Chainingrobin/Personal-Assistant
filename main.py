"""
Entry point / runtime loop for the Aegis assistant.

This file owns the full audio state machine:
wake word -> record utterance -> transcribe -> optional voice verification ->
orchestrate one command -> idle. The LLM orchestration itself still lives in
orchestrate.py and remains unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import AGENT_CONFIG
from orchestrate import AgentRequest, OllamaTransport, Orchestrator

from audio.stt import WhisperTranscriber
from audio.tts import PiperSpeaker
from audio.vad_recorder import VADRecorder
from audio.wake_word import WakeWordListener
from identity.intent_parser import extract_switch_user_intent
from identity.voice_id import (
    SpeakerVerifier,
    get_current_user,
    get_known_user_ids,
    switch_user,
)
from tools.add_calendar_event import add_calendar_event
from tools.auth import get_google_credentials
from tools.draft_email import draft_email
from tools.get_calendar_events import get_calendar_events
from tools.query_rag import query_rag
from tools.read_email import read_email
from identity.user_profiles import get_title

ALL_TOOLS = [read_email, get_calendar_events, add_calendar_event, draft_email]


# ---------------------------------------------------------------------------
# Mood detection — reads the user's own words so Aegis can answer in a
# matching tone (e.g. "add my sister's birthday party" -> excited).
# Keyword-based on purpose: no extra model, no added latency, easy to extend.
# ---------------------------------------------------------------------------

EXCITED_KEYWORDS = {
    "celebrate", "celebration", "birthday", "party", "wedding", "engagement",
    "congratulations", "congrats", "yay", "excited", "promotion", "promoted",
    "graduation", "graduate", "won", "win", "winning", "anniversary",
    "surprise party", "good news", "great news",
}

WARNING_KEYWORDS = {
    "help", "trouble", "emergency", "urgent", "problem", "broken", "fire",
    "danger", "scared", "worried", "stressed", "stress", "panic", "crisis",
    "hurry", "asap", "immediately",
}

SLOW_KEYWORDS = {
    "slowly", "slow down", "confused", "don't understand", "didn't understand",
    "explain again", "one more time", "repeat that", "say that again",
    "but slowly",
}


def detect_input_mood(user_input: str) -> str | None:
    """Look at the user's own phrasing and guess an emotional tone.
    Returns None if nothing matches, so callers can fall back to other logic."""
    lower = user_input.lower()

    if any(phrase in lower for phrase in EXCITED_KEYWORDS):
        return "excited"
    if any(phrase in lower for phrase in WARNING_KEYWORDS):
        return "warning"
    if any(phrase in lower for phrase in SLOW_KEYWORDS):
        return "slow"
    return None


def pick_mood(result, input_mood: str | None) -> str:
    """Decide the final speaking mood for a turn's response.

    Priority:
      1. A tool error always wins — the user must hear that something broke.
      2. The mood detected in the user's own request (celebration, distress,
         confusion) — this is what makes Aegis feel like it matches your tone.
      3. A successful create-type action (calendar/email) defaults to excited.
      4. Otherwise, normal.
    """
    if result.tool_result and "[ERROR]" in result.tool_result:
        return "warning"
    if input_mood:
        return input_mood
    if result.tool_called in ("add_calendar_event", "draft_email"):
        return "excited"
    return "normal"


def ensure_user_enrolled(user_id: str) -> None:
    print(f"Checking Google auth for {user_id}...", end=" ", flush=True)
    try:
        get_google_credentials(user_id)
        print("OK")
    except Exception as exc:
        print(f"FAILED ({exc})")
        raise


def build_system_prompt(user_id: str, rag_context: str = "") -> str:
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


def run_one_turn(orchestrator: Orchestrator, speaker: PiperSpeaker, user_id: str, user_input: str):
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
    result = orchestrator.run(request, current_user_id=user_id)

    input_mood = detect_input_mood(user_input)
    mood = pick_mood(result, input_mood)
    speaker.speak(result.raw_output, mood=mood)
    return result


@dataclass
class RuntimeComponents:
    orchestrator: Orchestrator
    speaker: PiperSpeaker
    wake_listener: WakeWordListener
    recorder: VADRecorder
    transcriber: WhisperTranscriber
    voice_verifier: SpeakerVerifier


def _handle_wake_event(runtime: RuntimeComponents) -> None:
    t0 = time.perf_counter()

    active_user = get_current_user()
    greeting = f"Hey {get_title(active_user)}, how can I help you today?"
    runtime.speaker.speak(greeting, mood="greeting")

    print("[state] RECORDING - capturing utterance...")
    utterance = runtime.recorder.record_utterance()
    t1 = time.perf_counter()
    print(f"[timing] VAD record: {t1 - t0:.2f}s")

    if utterance is None or utterance.samples.size == 0:
        print("[audio] No speech captured; returning to idle.")
        return

    print("[state] TRANSCRIBING...")
    transcript = runtime.transcriber.transcribe(utterance.samples, sample_rate=utterance.sample_rate).strip()
    t2 = time.perf_counter()
    print(f"[stt] {transcript}")
    print(f"[timing] Whisper STT: {t2 - t1:.2f}s")

    if not transcript:
        print("[stt] Empty transcript; returning to idle.")
        return

    claimed_user = extract_switch_user_intent(transcript, known_user_ids=get_known_user_ids())
    if claimed_user:
        print(f"[state] VERIFYING - switch intent for '{claimed_user}'...")
        t_verify0 = time.perf_counter()
        success, score = switch_user(
            claimed_user, utterance.samples, sample_rate=utterance.sample_rate, verifier=runtime.voice_verifier,
        )
        t_verify1 = time.perf_counter()
        print(f"[timing] ECAPA verify: {t_verify1 - t_verify0:.2f}s")
        if success:
            ensure_user_enrolled(claimed_user)
            print(f"[identity] Switched to {claimed_user} with similarity={score:.3f}")
        else:
            print(f"[identity] Voice verification failed for {claimed_user}; score={score:.3f}")
        return

    active_user = get_current_user()
    print(f"[state] DISPATCHING - active user '{active_user}'...")
    t3 = time.perf_counter()
    run_one_turn(runtime.orchestrator, runtime.speaker, active_user, transcript)
    t4 = time.perf_counter()
    print(f"[timing] Orchestrator (RAG+LLM+tools): {t4 - t3:.2f}s")
    print(f"[timing] TOTAL turn: {t4 - t0:.2f}s")

def main() -> None:
    print(f"[config] model={AGENT_CONFIG.model} profile={AGENT_CONFIG.hardware_profile} temp={AGENT_CONFIG.temperature}")
    transport = OllamaTransport(model=AGENT_CONFIG.model)
    orchestrator = Orchestrator(transport, max_retries=5, verbose=True)
    speaker = PiperSpeaker()

    voice_verifier = SpeakerVerifier()
    transcriber = WhisperTranscriber()
    recorder = VADRecorder()
    wake_listener = WakeWordListener()

    ensure_user_enrolled(get_current_user())

    runtime = RuntimeComponents(
        orchestrator=orchestrator,
        speaker=speaker,
        wake_listener=wake_listener,
        recorder=recorder,
        transcriber=transcriber,
        voice_verifier=voice_verifier,
    )

    print("Aegis is running. Say 'aegis' to wake it up. Press Ctrl+C to exit.")

    def _on_wake() -> None:
        _handle_wake_event(runtime)

    try:
        runtime.wake_listener.listen(_on_wake)
    except KeyboardInterrupt:
        print("\n[main] Exiting on user interrupt.")


if __name__ == "__main__":
    main()