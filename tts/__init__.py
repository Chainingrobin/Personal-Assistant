"""Pluggable local text-to-speech backends."""

from .base import AudioData, TTSBackend
from .factory import get_tts_backend
from .playback import play

__all__ = ["AudioData", "TTSBackend", "get_tts_backend", "play"]