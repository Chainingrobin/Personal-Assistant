"""Always-on wake-word detection for Aegis.
This module wraps openWakeWord, an open-source CPU-friendly wake-word engine.
Install it with `pip install openwakeword sounddevice`. The wake-word model is
loaded once when :class:`WakeWordListener` is constructed, and the listener then
runs continuously on the Raspberry Pi 5 CPU. openWakeWord caches its assets
locally on first use; custom wake-word models should be exported to ONNX and
placed at the path provided by `AEGIS_WAKE_WORD_MODEL` (default: `models/aegis.onnx`).
IMPORTANT: the InputStream is intentionally reopened after every callback
invocation rather than kept open across it. A command turn (VAD recording +
STT + LLM + tool call) can take 15-20+ seconds, during which nothing reads
from an open stream's buffer. Audio input streams buffer continuously at the
OS/driver level regardless of whether the code is reading them, so a stream
left open but unread across a long callback accumulates a backlog of stale
audio. The first reads after the callback returns then pull that backlog
instead of live audio, which can surface as a phantom wake detection even
though the user never said the wake word again. Reopening the stream after
each callback flushes that backlog. The one-time cost of reopening (roughly
tens to a couple hundred ms) is paid once per full command turn, not per
20ms frame, so it is negligible against the callback's own multi-second cost.

Pi note: USB audio adapters commonly support only 44100 Hz natively and reject
16000 Hz outright (PortAudio error -9997). The listener therefore opens the
InputStream at the device's reported default_samplerate and resamples each
captured frame down to the 16 kHz target before passing it to openwakeword.
"""
from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Callable
import numpy as np
try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover - import guard
    sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_ERROR = exc
else:
    _SOUNDDEVICE_ERROR = None
try:
    from openwakeword.model import Model as OpenWakeWordModel
except ImportError:  # pragma: no cover - import guard
    try:
        from openwakeword import Model as OpenWakeWordModel
    except ImportError as exc:  # pragma: no cover - import guard
        OpenWakeWordModel = None  # type: ignore[assignment]
        _OPENWAKEWORD_ERROR = exc
    else:
        _OPENWAKEWORD_ERROR = None
else:
    _OPENWAKEWORD_ERROR = None

from audio.resample import resample_frame


class WakeWordListener:
    """Continuously listen for the wake word and invoke a callback.
    Input: a zero-argument callback. Output: none; the callback runs when the
    wake score exceeds the configured threshold. The listener owns only the
    wake-word model; it never touches Whisper, ECAPA, or the LLM.
    """
    def __init__(
        self,
        model_paths: list[str] | None = None,
        sample_rate: int = 16000,
        frame_samples: int = 1600,
        detection_threshold: float = 0.5,
        device: int | None = None,
        debug: bool = True,
    ):
        if sd is None:
            raise ImportError("sounddevice is required for wake-word listening") from _SOUNDDEVICE_ERROR
        if OpenWakeWordModel is None:
            raise ImportError("openwakeword is required for wake-word listening") from _OPENWAKEWORD_ERROR
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.detection_threshold = detection_threshold
        self.device = device
        self.debug = debug
        self._stop_event = threading.Event()

        # Detect the device's native sample rate so we can open the InputStream
        # at a rate the hardware actually supports, then resample to target.
        device_info = sd.query_devices(device, "input") if sd is not None else {}
        self.device_rate = int(device_info.get("default_samplerate", sample_rate))
        # Scale frame size proportionally so each block still represents ~100ms
        self.device_frame_samples = int(round(frame_samples * self.device_rate / sample_rate))

        if self.debug and self.device_rate != self.sample_rate:
            print(
                f"[wake-debug] device native rate={self.device_rate} Hz, "
                f"target={self.sample_rate} Hz — resampling enabled"
            )

        resolved_paths = model_paths or self._resolve_model_paths()
        self.model_paths = [str(Path(path)) for path in resolved_paths]
        self.model = OpenWakeWordModel(wakeword_models=self.model_paths, inference_framework="onnx")

    def _resolve_model_paths(self) -> list[str]:
        """Resolve the local wake-word model path list.
        Output: a list of ONNX model paths. This keeps the runtime decoupled
        from training and lets each deployment ship its own custom model.
        """
        env_value = os.getenv("AEGIS_WAKE_WORD_MODEL", "models/aegis.onnx")
        paths = [part.strip() for part in env_value.split(os.pathsep) if part.strip()]
        missing = [path for path in paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                "Wake-word model not found. Set AEGIS_WAKE_WORD_MODEL to a local ONNX file "
                f"or create the default path(s): {missing}"
            )
        return paths

    def stop(self) -> None:
        """Request that the listener loop exit on the next iteration."""
        self._stop_event.set()

    def _predict_scores(self, audio_chunk: np.ndarray) -> dict[str, float]:
        """Run openWakeWord on one audio chunk.
        Input: int16 mono audio at 16 kHz, shape (n,). Output: model score dict.
        On a Pi 5 this should remain lightweight because the model is tiny.
        """
        predictor = getattr(self.model, "predict", None)
        if callable(predictor):
            scores = predictor(audio_chunk)
        else:
            scores = self.model.predict_clip(audio_chunk)
        if isinstance(scores, dict):
            return {str(key): float(value) for key, value in scores.items()}
        return {"wake_word": float(np.max(np.asarray(scores, dtype=np.float32)))}

    def _reset_model_state(self) -> None:
        """Clear openWakeWord's internal rolling prediction buffer, if supported.
        openWakeWord keeps a short internal audio/feature buffer across calls
        to smooth its scoring. If that internal state isn't cleared when we
        reopen the mic stream, its own buffer (separate from the OS audio
        buffer) can also carry stale context into the first few predictions
        on a fresh stream. Not all openwakeword versions expose a reset
        method, so this is best-effort.
        """
        reset_fn = getattr(self.model, "reset", None)
        if callable(reset_fn):
            reset_fn()
            if self.debug:
                print("[wake-debug] model internal state reset")

    def listen(self, callback: Callable[[], None]) -> None:
        """Run the wake-word loop and call callback each time Aegis is detected.
        Input: zero-argument callback. Output: none; the method blocks until
        stop() is called or the process exits. The callback is invoked
        synchronously so it can safely launch recording without concurrent heavy
        inference. The mic stream is reopened after every callback to flush any
        audio buffered while the callback was running (see module docstring).
        """
        print("[state] IDLE — listening for wake word 'aegis'...")
        while not self._stop_event.is_set():
            self._reset_model_state()
            with sd.InputStream(
                samplerate=self.device_rate,
                channels=1,
                dtype="int16",
                blocksize=self.device_frame_samples,
                device=self.device,
            ) as stream:
                if self.debug:
                    print("[wake-debug] fresh input stream opened")
                while not self._stop_event.is_set():
                    frame, _ = stream.read(self.device_frame_samples)
                    raw_chunk = np.asarray(frame, dtype=np.int16).reshape(-1)
                    audio_chunk = resample_frame(raw_chunk, self.device_rate, self.sample_rate)
                    scores = self._predict_scores(audio_chunk)
                    best_name, best_score = max(scores.items(), key=lambda item: item[1])
                    if best_score >= self.detection_threshold:
                        print(f"[wake] Detected {best_name} at score={best_score:.3f}")
                        break
            if self._stop_event.is_set():
                break
            callback()
            print("[state] IDLE — listening for wake word 'aegis'...")