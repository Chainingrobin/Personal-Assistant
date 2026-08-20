"""Shared contracts for local text-to-speech backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioData:
    """Synthesized mono audio samples and their playback sample rate."""

    samples: np.ndarray
    sample_rate: int


class TTSBackend(ABC):
    """Interface implemented by each local TTS engine."""

    @abstractmethod
    def synthesize(self, text: str) -> AudioData:
        """Convert text to audio without playing it or writing a file."""
        raise NotImplementedError