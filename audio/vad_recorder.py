"""Microphone recording with VAD-based utterance end detection.

This module wraps the lightweight `webrtcvad` package and `sounddevice` input
streaming. Install with `pip install webrtcvad sounddevice`. The recorder uses
fixed 20 ms frames at 16 kHz, starts capturing after wake-word detection, and
returns once silence has lasted long enough to mark the end of the utterance.
No model downloads occur here; VAD is pure CPU and usually negligible on a Pi 5.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover - import guard
    sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_ERROR = exc
else:
    _SOUNDDEVICE_ERROR = None

try:
    import webrtcvad
except ImportError as exc:  # pragma: no cover - import guard
    webrtcvad = None  # type: ignore[assignment]
    _WEBRTCVAD_ERROR = exc
else:
    _WEBRTCVAD_ERROR = None


@dataclass(frozen=True)
class RecordedAudio:
    """Container for a captured utterance.

    samples: mono int16 PCM audio with shape (n,). sample_rate is always 16000.
    duration_seconds is the captured duration for logging and guard rails.
    """

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


class VADRecorder:
    """Record one utterance and stop when silence is detected.

    Input: none; the recorder owns the microphone stream. Output: a
    RecordedAudio object or None if no speech was captured. On a Pi 5 the VAD
    loop is cheap because it processes 20 ms frames with a small rule-based model.

    Speech hangover smoothing: a single noise-triggered "speech" frame during
    real silence (e.g. a fan hum blip) no longer resets the silence counter.
    Only `speech_hangover_frames` consecutive speech-flagged frames re-arm the
    silence counter reset, which makes end-of-utterance detection resistant to
    intermittent background noise while still responding quickly to real speech.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        silence_duration_seconds: float = 0.6,
        min_utterance_seconds: float = 2.0,
        max_utterance_seconds: float = 15.0,
        pre_roll_seconds: float = 0.5,
        aggressiveness: int = 3,
        speech_hangover_frames: int = 3,
        device: int | None = None,
        debug: bool = True,
        debug_interval_frames: int = 25,
    ):
        if sd is None:
            raise ImportError("sounddevice is required for microphone recording") from _SOUNDDEVICE_ERROR
        if webrtcvad is None:
            raise ImportError("webrtcvad is required for utterance segmentation") from _WEBRTCVAD_ERROR

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.silence_duration_seconds = silence_duration_seconds
        self.min_utterance_seconds = min_utterance_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.pre_roll_frames = max(1, int(pre_roll_seconds * 1000 / frame_duration_ms))
        self.speech_hangover_frames = speech_hangover_frames
        self.device = device
        self.debug = debug
        self.debug_interval_frames = debug_interval_frames
        self.vad = webrtcvad.Vad(aggressiveness)

    def _read_frame(self, stream: sd.InputStream) -> np.ndarray:
        """Read one 20 ms frame from the microphone as mono int16 PCM."""
        frame, _ = stream.read(self.frame_samples)
        frame = np.asarray(frame, dtype=np.int16)
        if frame.ndim == 2:
            frame = frame[:, 0]
        return np.ascontiguousarray(frame.reshape(-1))

    def record_utterance(self) -> RecordedAudio | None:
        """Record until silence marks the end of the utterance.

        Output: mono int16 PCM buffer or None if the user never spoke. This call
        usually runs for 3-15 seconds on the Pi depending on the utterance length.
        """
        if sd is None:
            raise ImportError("sounddevice is required for microphone recording") from _SOUNDDEVICE_ERROR

        t_start = time.perf_counter()
        recorded_frames: list[bytes] = []
        pre_roll: Deque[bytes] = deque(maxlen=self.pre_roll_frames)
        speech_started = False
        speech_frames = 0
        silence_frames = 0
        speech_hangover_counter = 0
        frames_since_speech_start = 0
        max_frames = int(self.max_utterance_seconds * 1000 / self.frame_duration_ms)
        silence_limit = int(self.silence_duration_seconds * 1000 / self.frame_duration_ms)
        # Minimum ELAPSED frames since speech started before we allow an early
        # stop — NOT a count of VAD-flagged speech frames. Gating on flagged
        # speech-frame count is unreliable: short utterances or a strict VAD
        # aggressiveness setting can flag far fewer "speech" frames than the
        # utterance's real duration, which previously prevented the silence
        # break condition from ever firing.
        min_elapsed_frames = int(self.min_utterance_seconds * 1000 / self.frame_duration_ms)

        if self.debug:
            print(
                f"[vad-debug] starting: silence_limit={silence_limit} frames "
                f"({self.silence_duration_seconds}s), min_elapsed_frames={min_elapsed_frames}, "
                f"hangover={self.speech_hangover_frames}"
            )

        hit_max_frames = False

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.device,
        ) as stream:
            for frame_index in range(max_frames):
                frame = self._read_frame(stream)
                frame_bytes = frame.tobytes()
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

                if not speech_started:
                    pre_roll.append(frame_bytes)
                    if is_speech:
                        speech_started = True
                        recorded_frames.extend(pre_roll)
                        pre_roll.clear()
                        recorded_frames.append(frame_bytes)
                        speech_frames += 1
                        speech_hangover_counter = 1
                        if self.debug:
                            print(f"[vad-debug] speech started at frame {frame_index}")
                    continue

                recorded_frames.append(frame_bytes)
                frames_since_speech_start += 1

                if is_speech:
                    speech_frames += 1
                    speech_hangover_counter += 1
                    # Only reset silence once we've seen enough consecutive
                    # speech-flagged frames — filters out single-frame noise blips.
                    if speech_hangover_counter >= self.speech_hangover_frames:
                        silence_frames = 0
                else:
                    speech_hangover_counter = 0
                    silence_frames += 1

                if self.debug and frame_index % self.debug_interval_frames == 0:
                    elapsed = frame_index * self.frame_duration_ms / 1000.0
                    print(
                        f"[vad-debug] t={elapsed:.2f}s is_speech={is_speech} "
                        f"speech_frames={speech_frames} silence_frames={silence_frames}/{silence_limit} "
                        f"elapsed_since_speech={frames_since_speech_start}/{min_elapsed_frames}"
                    )

                if frames_since_speech_start >= min_elapsed_frames and silence_frames >= silence_limit:
                    if self.debug:
                        print(f"[vad-debug] silence threshold reached at frame {frame_index}; ending utterance")
                    break
            else:
                hit_max_frames = speech_started

        if hit_max_frames and self.debug:  # noqa: SIM102 - kept for clarity
            print("[vad-debug] WARNING: hit max_utterance_seconds cap without detecting silence — "
                  "check for background noise, or raise aggressiveness / speech_hangover_frames")

        if not recorded_frames:
            if self.debug:
                print(f"[vad-debug] no speech captured ({time.perf_counter() - t_start:.2f}s elapsed)")
            return None

        audio = np.frombuffer(b"".join(recorded_frames), dtype=np.int16).copy()
        duration_seconds = float(audio.shape[0]) / float(self.sample_rate)
        if self.debug:
            print(
                f"[vad-debug] utterance complete: {duration_seconds:.2f}s audio, "
                f"{time.perf_counter() - t_start:.2f}s wall time"
            )
        return RecordedAudio(samples=audio, sample_rate=self.sample_rate, duration_seconds=duration_seconds)