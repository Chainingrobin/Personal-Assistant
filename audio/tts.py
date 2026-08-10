"""Local, offline text-to-speech synthesis for Aegis using Piper."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    sd = None
    _SOUNDDEVICE_ERROR = exc
else:
    _SOUNDDEVICE_ERROR = None

try:
    from piper import PiperVoice
except ImportError as exc:
    PiperVoice = None
    _PIPER_ERROR = exc
else:
    _PIPER_ERROR = None


DEFAULT_MODEL_PATH = "models/tts/en_US-lessac-low.onnx"


class PiperSpeaker:
    """Load a Piper voice once and synthesize/play text on demand."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device=None, debug=True):
        if sd is None:
            raise ImportError("sounddevice is required for audio playback") from _SOUNDDEVICE_ERROR
        if PiperVoice is None:
            raise ImportError("piper-tts is required for speech synthesis") from _PIPER_ERROR

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Piper voice model not found at {model_path}")

        self.model_path = model_path
        self.device = device
        self.debug = debug
        self.voice = PiperVoice.load(str(model_path))

    def _synthesize_to_wav_bytes(self, text):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        return buffer.getvalue()

    def synthesize(self, text):
        wav_bytes = self._synthesize_to_wav_bytes(text)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            raw_audio = wav_file.readframes(n_frames)
        samples = np.frombuffer(raw_audio, dtype=np.int16)
        return samples, sample_rate

    def speak(self, text):
        text = text.strip()
        if not text:
            if self.debug:
                print("[tts-debug] empty text, skipping playback")
            return

        if self.debug:
            print("[tts-debug] synthesizing " + str(len(text)) + " chars")

        samples, sample_rate = self.synthesize(text)

        if self.debug:
            duration = samples.shape[0] / float(sample_rate)
            print("[tts-debug] playing " + str(round(duration, 2)) + "s audio")

        sd.play(samples, samplerate=sample_rate, device=self.device)
        sd.wait()
