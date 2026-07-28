"""
Entry point / runtime loop for the Jarvis assistant.

This file owns the LIFECYCLE (wake -> identify -> converse -> respond -> repeat).
orchestrate.py owns a single request/response cycle and knows nothing about
who the user is or how they were identified — that separation is deliberate,
so swapping identity methods later never requires touching orchestrate.py.
"""
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

ALL_TOOLS = [read_email, get_calendar_events, add_calendar_event, draft_email, query_rag]


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



def build_system_prompt(user_id: str) -> str:
    """Build the system prompt for the current turn.

    FUTURE: this is where per-user RAG context gets pulled in and injected,
    e.g. retriever.retrieve(query, user_id=user_id) -> format_context(...).
    For now it's a static string with the user's name swapped in, just to
    prove the LLM's behavior visibly changes when the active user changes.
    """
    return (
        f"You are Aegis, a personal assistant serving {user_id}.\n"
        "RULES:\n"
        "1. If the user asks for data from email, calendar, or knowledge base, you MUST call a tool directly. NEVER write text promising to search or check—call the tool immediately.\n"
        "2. If the user prompt is a greeting, general chit-chat, or follow-up question that doesn't need external data, reply directly with plain text without tools.\n"
        "3. Never ask for confirmation before calling a read-only tool."
    )


def run_one_turn(orchestrator: Orchestrator, user_id: str, user_input: str):
    """Run a single conversational turn for the given identified user."""
    request = AgentRequest(
        messages=[
            {"role": "system", "content": build_system_prompt(user_id)},
            {"role": "user", "content": user_input},
        ],
        tools=ALL_TOOLS,
    )
    # current_user_id flows into orchestrate.run(), which injects it into
    # tool args after the LLM picks a tool — the model itself never sees it.
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

        result = run_one_turn(orchestrator, get_current_user(), raw)
        print(result.raw_output)


if __name__ == "__main__":
    main()