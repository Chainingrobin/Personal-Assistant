"""Per-user Google OAuth helpers.

Deployment note:
- On a laptop or other machine with a browser, the first run may fall back to
  the interactive consent flow.
- On the headless Raspberry Pi, that branch should never run. Tokens must
  already exist under credentials/tokens/<user_id>.json and be copied over
  after one-time enrollment on a laptop with a browser.

This stays intentionally simple: one JSON token file per user, no database,
no encryption layer, and no extra abstraction.
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from config import GOOGLE_CONFIG


def _token_path(user_id: str) -> Path:
    if not user_id:
        raise ValueError("user_id is required")
    normalized = user_id.replace("_", "").replace("-", "")
    if not normalized.isalnum():
        raise ValueError(f"Invalid user_id: {user_id!r}")
    return Path(GOOGLE_CONFIG.token_dir) / f"{user_id}.json"


def get_google_credentials(user_id: str) -> Credentials:
    token_path = _token_path(user_id)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    credentials_path = Path(GOOGLE_CONFIG.credentials_path)
    if not credentials_path.exists():
        raise FileNotFoundError(f"Missing Google OAuth client file: {credentials_path}")

    creds: Credentials | None = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), scopes=list(GOOGLE_CONFIG.scopes))
        except ValueError:
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
        return creds

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes=list(GOOGLE_CONFIG.scopes))
    creds = flow.run_local_server(host="127.0.0.1", port=0)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds
