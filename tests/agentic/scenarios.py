from .schemas import SyntheticCase


def build_cases() -> list[SyntheticCase]:
    return [
        SyntheticCase(
            case_id="mail_meeting_request",
            task_type="mail_parsing",
            prompt=(
                "Parse this email into JSON with sender, intent, priority, and action_items:\n"
                "From: alex@example.com\n"
                "Subject: Quick sync tomorrow\n"
                "Body: Can we meet tomorrow at 3pm to review the launch plan?"
            ),
            expected={
                "sender": "alex@example.com",
                "intent": "meeting_request",
                "priority": "normal",
            },
        ),
        SyntheticCase(
            case_id="calendar_event_extraction",
            task_type="calendar_parsing",
            prompt=(
                "Extract a calendar event from this text and return JSON with title, start, end, and attendees:\n"
                "Team sync on 2026-08-02 from 10:00 to 10:30 with sam@example.com and priya@example.com."
            ),
            expected={
                "title": "Team sync",
                "start": "2026-08-02T10:00:00",
                "end": "2026-08-02T10:30:00",
            },
        ),
        SyntheticCase(
            case_id="event_logging",
            task_type="event_logging",
            prompt=(
                "Return JSON for a log event with type, severity, and summary:\n"
                "User opened the assistant, read two emails, then created a calendar draft."
            ),
            expected={
                "type": "assistant_activity",
                "severity": "info",
            },
        ),
    ]
