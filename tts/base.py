"""Shared contracts for local text-to-speech backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


import re

_MARKDOWN_STRIP_RE = re.compile(r"[*_`#~]|(\[.*?\]\(.*?\))")

def sanitize_for_speech(text: str) -> str:
    """Strip markdown formatting characters the model tends to emit,
    so TTS doesn't read '**' or '#' aloud. Applied right before synthesis,
    never before the text is displayed/logged."""
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)   # [label](url) -> label
    text = _MARKDOWN_STRIP_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

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