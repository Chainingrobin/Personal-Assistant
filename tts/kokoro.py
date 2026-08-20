"""Kokoro ONNX text-to-speech backend."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from config import TTS_CONFIG, TTSConfig

from .base import AudioData, TTSBackend

try:
    Kokoro = import_module("kokoro_onnx").Kokoro
except ImportError as exc:  # pragma: no cover - depends on optional install
    Kokoro = None
    _KOKORO_ERROR = exc
else:
    _KOKORO_ERROR = None


class KokoroBackend(TTSBackend):
    """Synthesize audio with the local Kokoro ONNX model."""

    def __init__(self, config: TTSConfig = TTS_CONFIG):
        if Kokoro is None:
            raise ImportError("kokoro-onnx is required for the Kokoro TTS backend") from _KOKORO_ERROR

        model_path = Path(config.model_path)
        voices_path = Path(config.voices_path)
        missing = [str(path) for path in (model_path, voices_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Kokoro model files are missing: " + ", ".join(missing) +
                ". Run scripts/download_tts_models.py."
            )

        self.voice = config.voice
        self.speed = config.speed
        self._kokoro = Kokoro(str(model_path), str(voices_path))

    def synthesize(self, text: str) -> AudioData:
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        samples, sample_rate = self._kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return AudioData(samples=samples, sample_rate=int(sample_rate))