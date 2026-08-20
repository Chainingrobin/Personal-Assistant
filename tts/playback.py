"""Synchronous audio playback for synthesized TTS output."""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from .base import AudioData


def play(audio: AudioData) -> None:
    """Play audio and block until the device has finished consuming it."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    sd.play(samples, audio.sample_rate)
    sd.wait()