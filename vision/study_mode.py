"""Study-mode monitor for head-pose-based procrastination detection."""
from __future__ import annotations
import cv2
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional output path
    sd = None  # type: ignore[assignment]

from config import VisionConfig
from .camera_backend import CameraBackend, get_camera_backend
from .display import close_debug_window, get_headless_preview_path, render_debug_frame, show_debug_frame
from .head_pose import HeadPoseDetection, HeadPoseDetector

EventKind = Literal["distraction", "escalation"]


@dataclass(frozen=True)
class StudyModeEvent:
    kind: EventKind
    timestamp: datetime
    yaw_deg: float
    pitch_deg: float
    away_duration_sec: float
    rolling_window_count: int


def _timestamp_label(timestamp: datetime | None = None) -> str:
    timestamp = timestamp or datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _play_notification_sound() -> None:
    if sd is None:
        return
    sample_rate = 44100
    duration_sec = 0.25
    frequency_hz = 880.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = (0.16 * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)
    try:
        sd.play(tone, sample_rate)
        sd.wait()
    except Exception:
        # Notification audio is best-effort; the distraction signal still logs.
        return


class StudyModeMonitor:
    """Monitor head pose and report distraction/escalation events.

    The monitor runs in its own worker thread, fully independent of the
    shared heavy-task lock used by wake-word/STT/voice-ID/LLM. Per-frame
    camera reads and MediaPipe inference are cheap (that's the whole point
    of choosing a pretrained, edge-optimized landmark model) and must NOT
    contend for the same lock those heavier operations use, or the
    wake-word thread ends up starved waiting for a lock this loop keeps
    re-acquiring ~10x/sec.

    The lock is only used at the one point that actually touches something
    heavy: when an escalation event is handed off to the LLM orchestrator.
    That handoff happens in the on_event callback (see main.py), not here,
    so this class stores the lock only to pass it through — it is never
    acquired inside the per-frame loop.
    """

    def __init__(
        self,
        config: VisionConfig,
        on_event: Callable[[StudyModeEvent], None] | None = None,
        heavy_task_lock: threading.Lock | None = None,
    ):
        self.config = config
        self.on_event = on_event
        # Stored for callers (e.g. main.py's escalation handler) that need
        # to serialize their own LLM calls against STT/voice-ID. Never
        # acquired directly inside this class's per-frame loop.
        self._heavy_task_lock = heavy_task_lock or threading.Lock()
        self._camera_backend: CameraBackend | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._window_name = "Aegis Study Mode"
        self._event_timestamps: deque[datetime] = deque()
        self._current_calibration: tuple[float, float] | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            print("[study] Study mode is already running.")
            return
        self._stop_event.clear()
        self._event_timestamps.clear()
        self._camera_backend = get_camera_backend(
            self.config.camera_backend,
            fps=self.config.camera_fps,
        )
        self._thread = threading.Thread(target=self._run, name="StudyModeMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._thread = None
        if self._camera_backend is not None:
            self._camera_backend.close()
            self._camera_backend = None
        close_debug_window(self._window_name)

    def _emit_event(self, event: StudyModeEvent) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception as exc:
            print(f"[study] Event handler failed: {exc}")

    def _log_distraction(self, detection: HeadPoseDetection, away_duration_sec: float) -> None:
        timestamp = _timestamp_label()
        print(
            f"[study][{timestamp}] distraction: yaw={detection.yaw_deg:.2f}° "
            f"pitch={detection.pitch_deg:.2f}° away_for={away_duration_sec:.2f}s"
        )
        _play_notification_sound()

    def _classify_frame(self, detection: HeadPoseDetection | None) -> tuple[str, HeadPoseDetection | None]:
        if detection is None:
            return "LOOKING AWAY", None
        assert self._current_calibration is not None
        baseline_yaw, baseline_pitch = self._current_calibration
        yaw_delta = abs(detection.yaw_deg - baseline_yaw)
        pitch_delta = abs(detection.pitch_deg - baseline_pitch)
        if yaw_delta <= self.config.yaw_tolerance_deg and pitch_delta <= self.config.pitch_tolerance_deg:
            return "SAFE ZONE", detection
        return "LOOKING AWAY", detection

    def _calibrate(self, detector: HeadPoseDetector) -> tuple[float, float]:
        print(
            f"[study] Calibrating for {self.config.calibration_duration_sec:.1f}s. "
            "Sit normally and look at the screen."
        )
        samples_yaw: list[float] = []
        samples_pitch: list[float] = []
        frames_seen = 0
        detections_seen = 0
        last_feedback_time = 0.0
        deadline = time.monotonic() + self.config.calibration_duration_sec
        frame_interval = 1.0 / max(1, self.config.camera_fps)
        next_frame_time = time.monotonic()
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_frame_time:
                time.sleep(min(frame_interval / 4.0, next_frame_time - now))
                continue

            # No heavy_task_lock here: camera read + MediaPipe inference is
            # cheap and must stay independent of STT/voice-ID/LLM so the
            # wake-word thread is never starved waiting on this loop.
            frame = self._camera_backend.read_frame() if self._camera_backend is not None else None
            frames_seen += 1

            # --- TEMP DIAGNOSTIC: dump the raw frame once, unconditionally ---
            if frames_seen == 1:
                if frame is None:
                    print("[diag] read_frame() returned None on first read")
                else:
                    print(f"[diag] frame shape={frame.shape} dtype={frame.dtype}")
                    cv2.imwrite("/tmp/aegis_raw_frame.jpg", frame)
                    print("[diag] wrote /tmp/aegis_raw_frame.jpg")
            # --- end diagnostic ---

            detection = detector.process_frame(frame) if frame is not None else None

            next_frame_time = time.monotonic() + frame_interval
            if detection is None:
                now = time.monotonic()
                if now - last_feedback_time >= 1.5:
                    last_feedback_time = now
                    print(
                        f"[study] Calibration sees no face yet ({frames_seen} frames checked, {detections_seen} detections). "
                        f"If GUI is unavailable, watch {get_headless_preview_path().resolve()} for the latest preview."
                    )
                continue
            detections_seen += 1
            samples_yaw.append(detection.yaw_deg)
            samples_pitch.append(detection.pitch_deg)
            if self.config.show_debug_window:
                calibration_note = f"Calibrating... samples={len(samples_yaw)}"
                debug_frame = render_debug_frame(frame, detection, "CALIBRATING", calibration_note)
                show_debug_frame(self._window_name, debug_frame)
        if len(samples_yaw) < 3:
            raise RuntimeError(
                "Calibration failed because too few face samples were collected. "
                "Make sure your face is visible and well lit during the calibration window. "
                f"If the GUI is unavailable, inspect {get_headless_preview_path().resolve()} for the camera preview."
            )
        # Median resists a single blink or transient head twitch better than the mean.
        baseline_yaw = float(np.median(samples_yaw))
        baseline_pitch = float(np.median(samples_pitch))
        print(
            f"[study] Calibration complete. baseline_yaw={baseline_yaw:.2f}° "
            f"baseline_pitch={baseline_pitch:.2f}°"
        )
        return baseline_yaw, baseline_pitch

    def _run(self) -> None:
        detector = HeadPoseDetector()
        away_start: float | None = None
        distraction_fired = False
        last_detection: HeadPoseDetection | None = None
        frame = None
        detection = None
        try:
            if self._camera_backend is None:
                raise RuntimeError("Study mode started without an initialized camera backend")
            self._camera_backend.open()
            self._current_calibration = self._calibrate(detector)
            frame_interval = 1.0 / max(1, self.config.camera_fps)
            next_frame_time = time.monotonic()
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_frame_time:
                    time.sleep(min(frame_interval / 4.0, next_frame_time - now))
                    continue

                # Same reasoning as _calibrate: no lock around per-frame work.
                # The lock is only relevant at escalation time, and that
                # handoff happens in the on_event callback (main.py), which
                # acquires it itself right before calling the orchestrator.
                frame = self._camera_backend.read_frame() if self._camera_backend is not None else None
                detection = detector.process_frame(frame) if frame is not None else None

                if detection is not None:
                    last_detection = detection
                next_frame_time = time.monotonic() + frame_interval
                status, visible_detection = self._classify_frame(detection)
                if status == "SAFE ZONE":
                    away_start = None
                    distraction_fired = False
                else:
                    if away_start is None:
                        away_start = time.monotonic()
                    away_duration_sec = time.monotonic() - away_start
                    if away_duration_sec >= self.config.distraction_duration_threshold_sec and not distraction_fired:
                        distraction_fired = True
                        baseline_yaw, baseline_pitch = self._current_calibration or (0.0, 0.0)
                        event_detection = visible_detection or last_detection or HeadPoseDetection(
                            yaw_deg=baseline_yaw,
                            pitch_deg=baseline_pitch,
                            roll_deg=0.0,
                            face_landmarks=None,
                            image_points=(),
                        )
                        self._log_distraction(event_detection, away_duration_sec)
                        event_time = datetime.now()
                        self._event_timestamps.append(event_time)
                        while self._event_timestamps and (
                            (event_time - self._event_timestamps[0]).total_seconds()
                            > self.config.escalation_window_sec
                        ):
                            self._event_timestamps.popleft()
                        distraction_event = StudyModeEvent(
                            kind="distraction",
                            timestamp=event_time,
                            yaw_deg=event_detection.yaw_deg,
                            pitch_deg=event_detection.pitch_deg,
                            away_duration_sec=away_duration_sec,
                            rolling_window_count=len(self._event_timestamps),
                        )
                        self._emit_event(distraction_event)
                        if len(self._event_timestamps) >= self.config.escalation_event_count:
                            escalation_event = StudyModeEvent(
                                kind="escalation",
                                timestamp=event_time,
                                yaw_deg=event_detection.yaw_deg,
                                pitch_deg=event_detection.pitch_deg,
                                away_duration_sec=away_duration_sec,
                                rolling_window_count=len(self._event_timestamps),
                            )
                            print(
                                f"[study][{_timestamp_label(event_time)}] escalation: "
                                f"{len(self._event_timestamps)} distractions in the last "
                                f"{self.config.escalation_window_sec:.0f}s"
                            )
                            self._emit_event(escalation_event)
                if self.config.show_debug_window:
                    calibration_note = ""
                    if self._current_calibration is not None:
                        calibration_note = (
                            f"baseline yaw={self._current_calibration[0]:.2f}° "
                            f"pitch={self._current_calibration[1]:.2f}° "
                            f"tol=±{self.config.yaw_tolerance_deg:.1f}/{self.config.pitch_tolerance_deg:.1f}°"
                        )
                    status_banner = "CALIBRATING" if self._current_calibration is None else status
                    if frame is not None:
                        debug_frame = render_debug_frame(frame, detection, status_banner, calibration_note)
                        show_debug_frame(self._window_name, debug_frame)
            print("[study] Stop requested; shutting down study mode monitor.")
        finally:
            detector.close()
            if self._camera_backend is not None:
                self._camera_backend.close()
            close_debug_window(self._window_name)