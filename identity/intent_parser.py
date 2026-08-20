"""Fuzzy intent parsing for switch-user commands.

This module is intentionally lightweight and pure-Python so it can be unit
tested without microphones or model weights. It extracts commands such as
"switch user robin", "switch to robin", or Whisper mis-transcriptions like
"switch user rob in" and maps them onto the nearest known user_id.

CHANGE LOG:
  - _clean_tail now strips punctuation from each word BEFORE checking it
    against the stopword set. Previously "please." (with a trailing period)
    never matched the stopword "please", so it survived filtering and
    contaminated the candidate string fed to alias/fuzzy matching.
  - Added a small set of filler words ("back", "again", "over", "now") that
    show up in real transcripts ("switch user back to X") but weren't in
    the original stopword list.
  - _alias_match changed from exact-equality to substring containment: the
    real-world transcripts this was built for come with unpredictable filler
    around the name ("back to usif", "usef, please"), and exact equality
    breaks the moment any surrounding word survives cleanup. Containment
    means "the alias appears in the cleaned candidate" rather than "the
    cleaned candidate IS the alias", which is far more robust to the actual
    variance seen in a live run.
  - Expanded _USER_ALIASES with variants observed in a real run ("usif",
    "use if") rather than only the ones anticipated beforehand -- treat this
    list as something to keep growing from real logs, not a one-time list.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Sequence

from identity.voice_id import get_known_user_ids

_STOPWORDS = {
    "user", "to", "please", "the", "my", "profile", "account", "voice",
    "back", "again", "over", "now",
}
_SWITCH_PATTERN = re.compile(r"\b(?:switch|change|swap)\b(?:\s+(?:user|account|profile|to))?\s+(?P<tail>.+)", re.IGNORECASE)
_STUDY_MODE_PATTERN = re.compile(r"\bstudy\s+mode\b", re.IGNORECASE)

# Known Whisper mis-transcriptions for names that don't match the STT model's
# phonetic/English priors well. Keep growing this from real logs -- Whisper
# doesn't converge on one consistent mis-transcription, it bounces between
# several near-misses ("usif", "use if", "usef", "usuf") across runs, so
# treat any new one seen in a live log as worth adding here.
#
# NOTE: some aliases here (e.g. "use it") are common English words/phrases on
# their own. This is only safe because _alias_match is only ever called on
# the tail AFTER _SWITCH_PATTERN has already matched a "switch/change/swap"
# command -- it does not scan arbitrary transcripts.
_USER_ALIASES: dict[str, tuple[str, ...]] = {
    "youssef": (
        "usuf", "usef", "youssef", "use f", "use it", "use if",
        "yousef", "yousif", "yusuf", "usif",
    ),
}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _clean_tail(tail: str) -> str:
    """Strip stopwords AND punctuation-contaminated stopwords from the tail.

    Each word has punctuation stripped before the stopword check, so
    "please." / "usif?" / "back," all get evaluated on their bare letters
    rather than surviving filtering just because of a trailing period,
    comma, or question mark.
    """
    words = []
    for raw_word in re.split(r"\s+", tail.lower().strip()):
        bare_word = re.sub(r"[^a-z0-9']+", "", raw_word)
        if not bare_word or bare_word in _STOPWORDS:
            continue
        words.append(bare_word)
    return " ".join(words)


def _alias_match(candidate_text: str, known_user_ids: Sequence[str]) -> str | None:
    """Check known systematic mis-transcriptions before falling back to fuzzy matching.

    Uses substring containment (alias appears within the cleaned candidate)
    rather than exact equality, so leftover filler words that _clean_tail
    doesn't know to strip (an unanticipated one-off word) don't silently
    break an otherwise-clear alias match.
    """
    normalized_candidate = _normalize(candidate_text)
    if not normalized_candidate:
        return None
    best_user = None
    best_alias_len = 0
    for user_id in known_user_ids:
        for alias in _USER_ALIASES.get(user_id, ()):
            normalized_alias = _normalize(alias)
            if not normalized_alias:
                continue
            if normalized_alias in normalized_candidate:
                # Prefer the longest matching alias if multiple could match,
                # to avoid a short alias accidentally winning over a more
                # specific/longer one.
                if len(normalized_alias) > best_alias_len:
                    best_user = user_id
                    best_alias_len = len(normalized_alias)
    return best_user


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

    Tries known alias mis-transcriptions first (_alias_match), then falls
    back to fuzzy string similarity (_best_match) for anything not already
    catalogued.
    """
    transcript = transcript.strip()
    if not transcript:
        return None
    known_user_ids = tuple(known_user_ids or get_known_user_ids())
    match = _SWITCH_PATTERN.search(transcript)
    if not match:
        return None
    candidate = _clean_tail(match.group("tail"))
    alias_hit = _alias_match(candidate, known_user_ids)
    if alias_hit:
        return alias_hit
    return _best_match(candidate, known_user_ids)


def extract_study_mode_intent(transcript: str) -> str | None:
    """Return 'enable' or 'disable' when the transcript is a study-mode command.

    The matching stays intentionally lightweight so the wake-word callback can
    decide control flow without involving the LLM. We only look for an explicit
    "study mode" phrase paired with a start/stop verb.
    """
    transcript = transcript.strip().lower()
    if not transcript or not _STUDY_MODE_PATTERN.search(transcript):
        return None
    enable_patterns = (
        r"\b(enable|start|turn on|turn study mode on|activate|begin|resume)\b",
        r"\bstudy mode\b.*\b(on|enable|activate|start|begin|resume)\b",
    )
    disable_patterns = (
        r"\b(disable|stop|turn off|turn study mode off|deactivate|end|pause)\b",
        r"\bstudy mode\b.*\b(off|disable|deactivate|stop|end|pause)\b",
    )
    if any(re.search(pattern, transcript) for pattern in enable_patterns):
        return "enable"
    if any(re.search(pattern, transcript) for pattern in disable_patterns):
        return "disable"
    return None