"""Fuzzy intent parsing for switch-user commands.

This module is intentionally lightweight and pure-Python so it can be unit
tested without microphones or model weights. It extracts commands such as
"switch user robin", "switch to robin", or Whisper mis-transcriptions like
"switch user rob in" and maps them onto the nearest known user_id.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Sequence

from identity.voice_id import get_known_user_ids

_STOPWORDS = {"user", "to", "please", "the", "my", "profile", "account", "voice"}
_SWITCH_PATTERN = re.compile(r"\b(?:switch|change|swap)\b(?:\s+(?:user|account|profile|to))?\s+(?P<tail>.+)", re.IGNORECASE)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _clean_tail(tail: str) -> str:
    words = [word for word in re.split(r"\s+", tail.lower().strip()) if word and word not in _STOPWORDS]
    return " ".join(words)


def _best_match(candidate_text: str, known_user_ids: Sequence[str]) -> str | None:
    if not candidate_text:
        return None

    normalized_candidate = _normalize(candidate_text)
    if not normalized_candidate:
        return None

    best_user = None
    best_score = 0.0
    for user_id in known_user_ids:
        normalized_user = _normalize(user_id)
        if not normalized_user:
            continue
        score = SequenceMatcher(None, normalized_candidate, normalized_user).ratio()
        if normalized_user in normalized_candidate or normalized_candidate in normalized_user:
            score = max(score, 0.95)
        if score > best_score:
            best_user = user_id
            best_score = score

    return best_user if best_score >= 0.72 else None


def extract_switch_user_intent(transcript: str, known_user_ids: Sequence[str] | None = None) -> str | None:
    """Return the most likely user_id if the transcript is a switch command.

    Input: one Whisper transcript string. Output: a matching user_id or None.
    The known user list defaults to the deployment's profile/user registry.
    """
    transcript = transcript.strip()
    if not transcript:
        return None

    known_user_ids = tuple(known_user_ids or get_known_user_ids())
    match = _SWITCH_PATTERN.search(transcript)
    if not match:
        return None

    candidate = _clean_tail(match.group("tail"))
    return _best_match(candidate, known_user_ids)
