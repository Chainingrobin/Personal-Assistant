from __future__ import annotations

from datetime import datetime, timedelta
from googleapiclient.discovery import build
from tools.auth import get_google_credentials


def get_calendar_events(start_date: str = "", end_date: str = "", *, user_id: str = "") -> str:
    """Retrieves events from the user's Google Calendar.

    Do NOT call this tool if the user is only asking what the date or time
    is right now — that has nothing to do with calendar events and must be
    answered directly from context with no tool call.

    Call this tool only when the user is asking about scheduled events,
    appointments, or plans for a specific day or range.

    DATE RANGE INFERENCE:
    - A single specific day → start_date=<that day>, end_date=''
    - "this week" → start_date=Monday of this week, end_date=Sunday of this week
    - A full named month → start_date=first day of that month, end_date=last day
    - "next week" → start_date=Monday of next week, end_date=Sunday of next week

    Args:
        start_date: Start date in 'YYYY-MM-DD' format. Required.
        end_date: End date in 'YYYY-MM-DD' format. Only set for multi-day
                  ranges (weeks, months); leave empty for a single day.
        user_id: Do not provide this argument. It is injected automatically.
    """
    if not user_id:
        raise ValueError("user_id is required")

    credentials = get_google_credentials(user_id)
    service = build("calendar", "v3", credentials=credentials, cache_discovery=False)

    now = datetime.now().astimezone()

    # --- PARSE START DATE ---
    if start_date and isinstance(start_date, str):
        try:
            start_dt = datetime.strptime(start_date.strip(), "%Y-%m-%d").astimezone()
        except ValueError:
            start_dt = now # Fallback to today on bad format
    else:
        start_dt = now

    # --- PARSE END DATE ---
    if end_date and isinstance(end_date, str):
        try:
            end_dt = datetime.strptime(end_date.strip(), "%Y-%m-%d").astimezone()
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            end_dt = start_dt.replace(hour=23, minute=59, second=59)
    else:
        # If no end date given, default to the end of the start_date
        end_dt = start_dt.replace(hour=23, minute=59, second=59)

    # Format for Google API
    time_min_str = start_dt.replace(hour=0, minute=0, second=0).isoformat() 
    time_max_str = end_dt.isoformat()

    response = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min_str,
            timeMax=time_max_str,
            singleEvents=True,
            orderBy="startTime",
            maxResults=25,
        )
        .execute()
    )

    events = response.get("items", [])
    
    # Create a nice label for the output
    label = start_dt.strftime("%Y-%m-%d")
    if start_dt.date() != end_dt.date():
        label += f" to {end_dt.strftime('%Y-%m-%d')}"

    if not events:
        return f"No calendar events found for {label}."

    lines = [f"Upcoming events for {label}:"]
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