"""Speech-to-text transcription for Aegis using faster-whisper.

Install with `pip install faster-whisper`. The first model load downloads the
selected Whisper checkpoint to the local Hugging Face/CTranslate2 cache. This
module defaults to `base.en` with `compute_type='int8'` for Raspberry Pi 5 CPU
use, and the model is loaded once when :class:`WhisperTranscriber` is created.
"""

from __future__ import annotations

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as exc:  # pragma: no cover - import guard
    WhisperModel = None  # type: ignore[assignment]
    _FASTER_WHISPER_ERROR = exc
else:
    _FASTER_WHISPER_ERROR = None


def _normalize_audio(audio_buffer: np.ndarray) -> np.ndarray:
    """Return mono float32 audio in the range [-1, 1]."""
    audio = np.asarray(audio_buffer)
    if audio.ndim == 2:
        audio = audio[:, 0]
    if audio.dtype.kind in {"i", "u"}:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32, copy=False)
    return np.ascontiguousarray(audio.reshape(-1))


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample 1D audio with linear interpolation for lightweight CPU use."""
    if source_rate == target_rate or audio.size == 0:
        return audio
    duration = audio.shape[0] / float(source_rate)
    target_count = max(1, int(duration * target_rate))
    source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


class WhisperTranscriber:
    """Load faster-whisper once and transcribe one utterance at a time.

    Input: mono audio buffer and sample_rate. Output: transcript string.
    On a Pi 5, `base.en` is a practical balance between speed and accuracy.
    """

    def __init__(self, model_name: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        if WhisperModel is None:
            raise ImportError("faster-whisper is required for transcription") from _FASTER_WHISPER_ERROR
        self.model_name = model_name
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe one utterance buffer to plain text.

        Input: int16 or float audio buffer plus sample_rate. Output: transcript.
        This call is the expensive STT step and is expected to take roughly 1-2 s
        for short commands on a Pi 5 CPU with `base.en`.
        """
        audio = _normalize_audio(audio_buffer)
        audio = _resample_linear(audio, sample_rate, 16000)
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcript = "".join(segment.text for segment in segments).strip()
        return transcript
