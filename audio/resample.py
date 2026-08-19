"""Shared audio resampling helper.

Converts captured device-rate audio down to the 16 kHz target rate that
VAD, openwakeword, and Whisper all expect. Using scipy.signal.resample_poly
with reduced up/down factors (via GCD) keeps the operation efficient on Pi.
"""
from __future__ import annotations
from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_frame(frame: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample a mono int16 audio frame from orig_rate to target_rate.

    Input:  int16 mono numpy array captured at orig_rate.
    Output: int16 mono numpy array at target_rate, clipped to int16 range.
    If orig_rate == target_rate the input is returned unchanged.
    """
    if orig_rate == target_rate:
        return frame
    g = gcd(orig_rate, target_rate)
    up = target_rate // g
    down = orig_rate // g
    resampled = resample_poly(frame.astype(np.float32), up, down)
    return np.clip(resampled, -32768, 32767).astype(np.int16)