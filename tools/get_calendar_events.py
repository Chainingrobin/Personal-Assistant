from __future__ import annotations

from datetime import datetime, timedelta

from googleapiclient.discovery import build

from tools.auth import get_google_credentials


# Google Calendar API events.list for the signed-in user's primary calendar.
def get_calendar_events(days_ahead: int = 7, *, user_id: str = "") -> str:
    """
    Retrieves upcoming calendar events within a given time window.

    Args:
        days_ahead: How many days ahead to look for events.
    """
    if not user_id:
        raise ValueError("user_id is required")

    credentials = get_google_credentials(user_id)
    service = build("calendar", "v3", credentials=credentials, cache_discovery=False)

    now = datetime.now().astimezone()
    response = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=(now + timedelta(days=days_ahead)).isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=10,
        )
        .execute()
    )

    events = response.get("items", [])
    if not events:
        return f"No calendar events found in the next {days_ahead} days."

    lines = [f"Upcoming events in the next {days_ahead} days:"]
    for event in events:
        start_info = event.get("start", {})
        start = start_info.get("dateTime") or start_info.get("date") or "unknown start"
        summary = event.get("summary", "Untitled event")
        location = event.get("location")
        line = f"- {start}: {summary}"
        if location:
            line += f" @ {location}"
        lines.append(line)

    return "\n".join(lines)