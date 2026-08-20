"""TTS backend selection."""

from __future__ import annotations

from config import TTS_CONFIG, TTSConfig

from .base import TTSBackend
from .kokoro import KokoroBackend


def get_tts_backend(config: TTSConfig = TTS_CONFIG) -> TTSBackend:
    """Create the configured backend; model loading happens at construction."""
    if config.backend == "kokoro":
        return KokoroBackend(config)
    raise ValueError(f"Unsupported TTS_BACKEND: {config.backend!r}")