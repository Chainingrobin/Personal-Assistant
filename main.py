"""
Entry point / runtime loop for the Jarvis assistant.

This file owns the LIFECYCLE (wake -> identify -> converse -> respond -> repeat).
orchestrate.py owns a single request/response cycle and knows nothing about
who the user is or how they were identified — that separation is deliberate,
so swapping identity methods later never requires touching orchestrate.py.
"""

from datetime import datetime, timedelta
from orchestrate import Orchestrator, OllamaTransport, AgentRequest
from config import AGENT_CONFIG

from tools.read_email import read_email
from tools.get_calendar_events import get_calendar_events
from tools.add_calendar_event import add_calendar_event
from tools.draft_email import draft_email
from tools.query_rag import query_rag
from tools.auth import get_google_credentials

# --- IDENTITY LAYER -----------------------------------------------------
# Phase 1 (now): manual, hardcoded user switching for testing multi-user
#                behavior without real sensors.
# Phase 2 (later): swap this single import line for identity.voice_id,
#                  which exposes the same get_current_user() interface.
# Phase 3 (later): swap again for identity.face_id, same interface again.
# main.py's loop below should not need to change across any of these phases.
from identity.manual import get_current_user, set_current_user, KNOWN_USERS

ALL_TOOLS = [read_email, get_calendar_events, add_calendar_event, draft_email]


def ensure_user_enrolled(user_id: str):
    """Check/refresh/enroll credentials for exactly one user — the one
    about to become active. Called at boot for the default user, and
    again whenever the active user changes via 'switch'. This is the
    only point where a browser popup can happen; on the Pi it should
    never fire because every KNOWN_USERS token already exists there."""
    print(f"Checking Google auth for {user_id}...", end=" ", flush=True)
    try:
        get_google_credentials(user_id)
        print("OK")
    except Exception as e:
        print(f"FAILED ({e})")
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


def run_one_turn(orchestrator: Orchestrator, user_id: str, user_input: str):
    # Only query RAG for personalization-flavored inputs.
    # Skip it for short date/time/email/calendar queries to avoid wasted
    # retrieval and the [RAG] print noise on every turn.
    SKIP_RAG_KEYWORDS = {"date", "time", "today", "tomorrow", "email", "emails", "calendar", "events", "schedule"}
    words = set(user_input.lower().split())
    should_query_rag = not words.intersection(SKIP_RAG_KEYWORDS)

    rag_context = ""
    if should_query_rag:
        rag_context = query_rag(query=user_input, top_k=3, user_id=user_id)

    request = AgentRequest(
        messages=[
            {"role": "system", "content": build_system_prompt(user_id, rag_context=rag_context)},
            {"role": "user", "content": user_input},
        ],
        tools=[read_email, get_calendar_events, add_calendar_event, draft_email],
        # query_rag removed — handled above, not a model-visible tool
    )
    return orchestrator.run(request, current_user_id=user_id)

def main():
    transport = OllamaTransport(model=AGENT_CONFIG.model)
    orchestrator = Orchestrator(transport, max_retries=5, verbose=True)

    ensure_user_enrolled(get_current_user())  # only the default active user

    print("Aegis is running. Type 'switch <user_id>' to change active user, 'quit' to exit.")
    while True:
        raw = input(f"[{get_current_user()}] > ").strip()
        if not raw:     #ignore empty text input to llm 
            continue
        if raw.lower() == "quit":
            break
        if raw.lower().startswith("switch "):
            new_user = raw.split(" ", 1)[1].strip()
            set_current_user(new_user)
            ensure_user_enrolled(new_user)  # check/enroll only the incoming user
            continue

        run_one_turn(orchestrator, get_current_user(), raw)

if __name__ == "__main__":
    main()