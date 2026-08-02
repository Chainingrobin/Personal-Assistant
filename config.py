from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _split_scopes(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default

    scopes = tuple(scope.strip() for scope in value.split(",") if scope.strip())
    return scopes or default


@dataclass(frozen=True)
class AgentConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60.0"))
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
    api_path: str = os.getenv("OLLAMA_API_PATH", "/api/chat")
    provider: str = os.getenv("OLLAMA_PROVIDER", "ollama")
    hardware_profile: str = os.getenv("HARDWARE_PROFILE", "desktop")


@dataclass(frozen=True)
class GoogleConfig:
    credentials_path: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials/credentials.json")
    token_dir: str = os.getenv("GOOGLE_TOKEN_DIR", "credentials/tokens")
    scopes: tuple[str, ...] = _split_scopes(
        os.getenv("GOOGLE_SCOPES"),
        (
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/calendar",
        ),
    )


AGENT_CONFIG = AgentConfig()
GOOGLE_CONFIG = GoogleConfig()
