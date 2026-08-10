"""Compatibility shim for the old manual identity switch path.

The runtime now uses identity.voice_id directly. This module keeps the old
imports working for any local scripts that still point at identity.manual.
"""

from __future__ import annotations

from identity.voice_id import DEFAULT_KNOWN_USERS as KNOWN_USERS
from identity.voice_id import get_current_user, set_current_user, switch_user
"""
Phase 1 identity resolution: manual, hardcoded user switching.

No sensors, no models — just an in-memory variable you flip yourself to
simulate different users being "detected." This exists purely to prove that
switching the active user actually changes tool results (different Gmail/
Calendar tokens) and RAG context, before any real face/voice recognition
is wired in.

Later phases (identity/voice_id.py, identity/face_id.py) must expose the
same two functions with the same signatures, so main.py never has to change
when you swap this out for real recognition.
"""

# Known demo users for this project. Each needs their own Google Cloud OAuth
# consent + their own token file at credentials/tokens/<user_id>.json.
KNOWN_USERS = ("youssef", "robin","mariam")

_current_user: str = "mariam"  # default active user at startup


def set_current_user(user_id: str) -> None:
    """Manually mark a different user as active. Stand-in for a future
    voice/face recognition event firing."""
    global _current_user
    if user_id not in KNOWN_USERS:
        print(f"[identity] Warning: '{user_id}' is not in KNOWN_USERS {KNOWN_USERS}")
    _current_user = user_id


def get_current_user() -> str:
    """Return whichever user is currently 'active'. Real phases will
    replace this body with an actual recognition result, same return type."""
    return _current_user