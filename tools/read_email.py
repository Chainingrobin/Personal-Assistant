from __future__ import annotations
import re
from googleapiclient.discovery import build
from tools.auth import get_google_credentials


def _header_value(headers: list[dict[str, str]], name: str) -> str:
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


_VALID_TIMEFRAME = re.compile(r"^\d+[dwmy]$")  # e.g. '1d', '7d', '2w', '1m', '1y'


def read_email(sender: str | None = None, timeframe: str | None = None, limit: int = 5, *, user_id: str = "") -> str:
    """
    Retrieves recent emails from the user's inbox.

    Do NOT invent a sender if the user didn't name one. Most queries like
    "what emails did I get today" have no sender at all — leave sender=None
    in that case. Only set sender when the user explicitly names a person,
    company, or domain (e.g. "emails from LinkedIn" -> sender='linkedin').

    Do NOT invent a timeframe if the user didn't specify one. Only set
    timeframe when the user mentions a time period (e.g. "today", "this
    week", "last month"). If the user just asks for "the last email from
    X" or "my latest email from X" with no time period mentioned, leave
    timeframe=None so the ENTIRE mailbox history is searched — do not
    default to '1d', since the most recent matching email may be older
    than a day.

    Args:
        user_id: Do not provide this argument. It is injected automatically.
        sender: Optional. Only set this if the user names a specific sender,
                domain, or company (e.g. 'linkedin', 'github.com'). Leave as
                None for general queries like "today's emails" or "my inbox".
        timeframe: Optional. Only set this if the user mentions a specific
                   time period. Must be a relative value in this exact
                   format: '1d' = last day, '7d' = last week, '1m' = last
                   month, '1y' = last year. NEVER pass a literal calendar
                   date like '2026-07-29'. Leave as None if no time period
                   was mentioned — searching all history is the safe default,
                   not the last day.

    EXAMPLES:
    "what emails did I get today" -> sender=None, timeframe='1d'
    "emails from LinkedIn this week" -> sender='linkedin', timeframe='7d'
    "the last email I got from youtube" -> sender='youtube', timeframe=None
    "did I get anything from snapchat" -> sender='snapchat', timeframe=None
    """
    if not user_id:
        raise ValueError("user_id is required")

    # Guard against malformed timeframe values (e.g. a literal date) instead
    # of silently building a query Gmail will reject.
    if timeframe and not _VALID_TIMEFRAME.match(timeframe):
        return (
            f"[ERROR] Invalid timeframe '{timeframe}'. Expected a relative "
            f"value like '1d', '7d', '1m', or '1y'."
        )

    credentials = get_google_credentials(user_id)
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    query_parts = []
    if sender:
        query_parts.append(f"from:{sender}")
    if timeframe:
        query_parts.append(f"newer_than:{timeframe}")

    query = " ".join(query_parts) if query_parts else None

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