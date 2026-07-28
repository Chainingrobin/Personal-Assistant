from __future__ import annotations

from googleapiclient.discovery import build
from tools.auth import get_google_credentials


def _header_value(headers: list[dict[str, str]], name: str) -> str:
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


# Gmail API users.messages.list and users.messages.get for the signed-in mailbox.
def read_email(sender: str | None = None, timeframe: str | None = None, limit: int = 5, *, user_id: str = "") -> str:
    """
    Retrieves recent emails from the user's inbox.
    
    Args:
        sender: Optional. The specific email address or domain to filter by (e.g., 'youtube').
        timeframe: Optional. Time filter. Use '1d' for last day, '1w' for last week, '1m' for last month, '1y' for last year.
        limit: The maximum number of emails to return. Protects context window size.
    """    
    if not user_id:
        raise ValueError("user_id is required")

    credentials = get_google_credentials(user_id)
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    # Build the Gmail search string
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