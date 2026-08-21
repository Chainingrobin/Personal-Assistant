"""Synchronous audio playback for synthesized TTS output."""

from __future__ import annotations

from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from .base import AudioData


def _get_output_sample_rate(fallback: int) -> int:
    """Return the selected output device's default sample rate when available."""
    try:
        default_device = getattr(getattr(sd, "default", None), "device", None)
        device_index = 0
        if isinstance(default_device, (list, tuple)) and default_device:
            device_index = int(default_device[0])
        elif isinstance(default_device, (int, float)):
            device_index = int(default_device)

        devices = sd.query_devices()
        if hasattr(devices, "__len__") and len(devices):
            if 0 <= device_index < len(devices):
                device_info = devices[device_index]
            else:
                device_info = devices[0]
            if isinstance(device_info, dict):
                rate = device_info.get("default_samplerate")
                if rate is not None:
                    return int(rate)

        if isinstance(devices, dict):
            rate = devices.get("default_samplerate")
            if rate is not None:
                return int(rate)

        device_info = sd.query_devices(kind="output")
        if isinstance(device_info, dict):
            rate = device_info.get("default_samplerate")
            if rate is not None:
                return int(rate)
        if hasattr(device_info, "__len__") and len(device_info):
            first = device_info[0]
            if isinstance(first, dict):
                rate = first.get("default_samplerate")
                if rate is not None:
                    return int(rate)
    except Exception:
        pass
    return fallback


def _resample_to_supported_rate(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono float32 audio to the output device sample rate with scipy."""
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)

    g = gcd(source_rate, target_rate)
    up = target_rate // g
    down = source_rate // g
    resampled = resample_poly(np.asarray(samples, dtype=np.float32), up, down)
    return np.clip(resampled, -1.0, 1.0).astype(np.float32)


def play(audio: AudioData) -> None:
    """Play audio and block until the device has finished consuming it."""
    target_rate = _get_output_sample_rate(int(audio.sample_rate))
    samples = _resample_to_supported_rate(audio.samples, int(audio.sample_rate), target_rate)
    sd.play(samples, target_rate)
    sd.wait()