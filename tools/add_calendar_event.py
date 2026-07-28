from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, time as time_type, timedelta

from googleapiclient.discovery import build

from tools.auth import get_google_credentials


# Google Calendar API events.insert for the signed-in user's primary calendar.
def add_calendar_event(
    title: str,
    date: str,
    time: str = "12:00",
    duration_minutes: int = 60,
    *,
    user_id: str = "",
) -> str:

    """
    Adds a new event to the user's calendar. You MUST extract the event title
    and date directly from the user's message — never call this with empty arguments.

    Args:
        title: The exact name/title of the event as mentioned by the user.
        date: The date of the event in YYYY-MM-DD format. Infer relative dates
              (e.g. "tomorrow") from context if no explicit date is given.
        time: The time of the event in HH:MM 24-hour format. Defaults to 12:00 if unspecified.
        duration_minutes: Duration in minutes. Defaults to 60 if unspecified.
    """

    
    if not user_id:
        raise ValueError("user_id is required")

    credentials = get_google_credentials(user_id)
    service = build("calendar", "v3", credentials=credentials, cache_discovery=False)

    start_date = date_type.fromisoformat(date)
    start_time = time_type.fromisoformat(time)
    start = datetime.combine(start_date, start_time).astimezone()
    end = start + timedelta(minutes=duration_minutes)

    event = {
        "summary": title,
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()},
    }

    created = service.events().insert(calendarId="primary", body=event).execute()
    event_link = created.get("htmlLink", "")
    return f"Created calendar event '{title}' on {date} at {time} for {duration_minutes} minutes. {event_link}".strip()