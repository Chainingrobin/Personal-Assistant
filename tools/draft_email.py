from __future__ import annotations

import base64
from email.message import EmailMessage

from googleapiclient.discovery import build

from tools.auth import get_google_credentials


# Gmail API drafts.create for a draft-only email, never send.
def draft_email(recipient: str, subject: str, body: str, *, user_id: str = "") -> str:
    """
    Drafts an email ready to be sent via the Gmail API.

    Args:
        recipient: The email address of the recipient.
        subject: The subject line of the email.
        body: The full body text of the email.
    """
    if not user_id:
        raise ValueError("user_id is required")

    credentials = get_google_credentials(user_id)
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    draft = service.users().drafts().create(userId="me", body={"message": {"raw": raw_message}}).execute()
    draft_id = draft.get("id", "unknown")
    return f"Draft created for {recipient} with subject '{subject}'. Draft ID: {draft_id}"