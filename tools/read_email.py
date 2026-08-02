from __future__ import annotations
from datetime import datetime, timedelta
import dateparser
from googleapiclient.discovery import build
from tools.auth import get_google_credentials


def _header_value(headers: list[dict[str, str]], name: str) -> str:
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def _resolve_timeframe(phrase: str) -> tuple[str, str] | None:
    """
    Resolves a natural language time phrase (e.g. "today", "last weekend",
    "3 days ago") into (after, before) date strings in Gmail query format
    (YYYY/MM/DD). Returns None if the phrase can't be parsed.

    Anchors on a calendar day: the resolved window is midnight-to-midnight
    for the day dateparser resolves the phrase to. For ranges naturally
    wider than a day (e.g. "this week"), before is left open-ended.
    """
    settings = {
        "PREFER_DATES_FROM": "past",
        "RELATIVE_BASE": datetime.now(),
    }
    parsed = dateparser.parse(phrase, settings=settings)
    if parsed is None:
        return None

    key = phrase.strip().lower()
    day_start = parsed.replace(hour=0, minute=0, second=0, microsecond=0)

    # Wider-than-a-day phrasing: anchor "after" but leave "before" open
    # (i.e. up to now) rather than boxing it into a single calendar day.
    wide_range_markers = ("week", "month", "year")
    if any(marker in key for marker in wide_range_markers):
        return day_start.strftime("%Y/%m/%d"), ""

    # Calendar-day phrasing ("today", "yesterday", a specific date, etc.):
    # tight midnight-to-midnight window.
    day_end = day_start + timedelta(days=1)
    return day_start.strftime("%Y/%m/%d"), day_end.strftime("%Y/%m/%d")


def read_email(sender: str | None = None, timeframe: str | None = None, limit: int = 5, *, user_id: str = "") -> str:
    """
    Retrieves recent emails from the user's inbox.

    Do NOT invent a sender if the user didn't name one. Most queries like
    "what emails did I get today" have no sender at all — leave sender=None
    in that case. Only set sender when the user explicitly names a person,
    company, or domain (e.g. "emails from LinkedIn" -> sender='linkedin').

    Do NOT invent a timeframe if the user didn't specify one. Only set
    timeframe when the user mentions a time period. Pass it through
    exactly as the user phrased it (e.g. "today", "yesterday", "last
    weekend", "this week", "3 days ago") — do NOT convert it to any
    special format yourself, date parsing happens internally. If the
    user just asks for "the last email from X" or "my latest email from
    X" with no time period mentioned, leave timeframe=None so the ENTIRE
    mailbox history is searched — do not default to "today".

    Args:
        user_id: Do not provide this argument. It is injected automatically.
        sender: Optional. Only set this if the user names a specific sender,
                domain, or company (e.g. 'linkedin', 'github.com'). Leave as
                None for general queries like "today's emails" or "my inbox".
        timeframe: Optional. Only set this if the user mentions a specific
                   time period. Pass the phrase as-is, in natural language.
                   Leave as None if no time period was mentioned.

    EXAMPLES:
    "what emails did I get today" -> sender=None, timeframe='today'
    "emails from LinkedIn this week" -> sender='linkedin', timeframe='this week'
    "what did I get last weekend" -> sender=None, timeframe='last weekend'
    "the last email I got from youtube" -> sender='youtube', timeframe=None
    "did I get anything from snapchat" -> sender='snapchat', timeframe=None
    """
    if not user_id:
        raise ValueError("user_id is required")

    query_parts = []
    if sender:
        query_parts.append(f"from:{sender}")

    if timeframe:
        resolved = _resolve_timeframe(timeframe)
        if resolved is None:
            return (
                f"[ERROR] Couldn't understand timeframe '{timeframe}'. "
                f"Try phrasing like 'today', 'this week', or 'last month'."
            )
        after, before = resolved
        query_parts.append(f"after:{after}")
        if before:
            query_parts.append(f"before:{before}")

    query = " ".join(query_parts) if query_parts else None

    credentials = get_google_credentials(user_id)
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    response = service.users().messages().list(userId="me", maxResults=limit, q=query).execute()
    messages = response.get("messages", [])
    if not messages:
        return "No emails found matching the criteria."

    lines = [f"Found {len(messages)} email(s) (showing up to {limit}):"]
    for message in messages:
        message_data = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        headers = message_data.get("payload", {}).get("headers", [])
        from_value = _header_value(headers, "From") or "Unknown sender"
        subject = _header_value(headers, "Subject") or "(no subject)"
        date = _header_value(headers, "Date") or "unknown date"
        snippet = message_data.get("snippet", "").strip()
        lines.append(f"- From: {from_value} | Subject: {subject} | Date: {date} | Snippet: {snippet}")
    return "\n".join(lines)