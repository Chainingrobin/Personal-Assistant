"""Camera backend abstraction for study mode.

The rest of the vision pipeline talks only to this module. That matters on a
Raspberry Pi 5 because the native CSI camera stack is libcamera-based and is
best accessed through Picamera2, while laptop/webcam development is simpler
through OpenCV's VideoCapture. Keeping both paths behind one interface avoids
letting the detection code grow backend-specific branches.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
import numpy as np

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _CV2_ERROR = exc
else:
    _CV2_ERROR = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

if TYPE_CHECKING:
    # Only imported for type hints; never executed at runtime, so it doesn't
    # care whether these are actually installed on this machine.
    import cv2 as cv2_types
    from picamera2 import Picamera2 as Picamera2Type


logger = logging.getLogger(__name__)


class CameraBackend(ABC):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 10):
        self.width = width
        self.height = height
        self.requested_fps = fps
        self._resolution: tuple[int, int] | None = None
        self._fps: float | None = None

    @abstractmethod
    def open(self) -> "CameraBackend":
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @property
    def resolution(self) -> tuple[int, int] | None:
        return self._resolution

    @property
    def fps(self) -> float | None:
        return self._fps


class WebcamBackend(CameraBackend):
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 10):
        super().__init__(width=width, height=height, fps=fps)
        self.device_index = device_index
        self._capture: Optional["cv2_types.VideoCapture"] = None

    def open(self) -> "WebcamBackend":
        if cv2 is None:
            raise ImportError("opencv-python is required for webcam camera access") from _CV2_ERROR

        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open webcam device index {self.device_index}. "
                "If this is a USB webcam, check permissions and that no other app is using it."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        capture.set(cv2.CAP_PROP_FPS, float(self.requested_fps))
        self._capture = capture
        self._resolution = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width,
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height,
        )
        self._fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(self.requested_fps)
        return self

    def read_frame(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        success, frame = self._capture.read()
        if not success:
            return None
        self._resolution = (int(frame.shape[1]), int(frame.shape[0]))
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class PiCameraBackend(CameraBackend):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 10):
        super().__init__(width=width, height=height, fps=fps)
        self._camera: Optional["Picamera2Type"] = None

    def open(self) -> "PiCameraBackend":
        if Picamera2 is None:
            raise ImportError(
                "picamera2 is required for the Pi camera backend. Install the Pi-only extras on the Raspberry Pi."
            )

        camera = Picamera2()
        video_config = camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        camera.configure(video_config)
        camera.start()
        self._camera = camera
        self._resolution = (self.width, self.height)
        self._fps = float(self.requested_fps)
        return self

    def read_frame(self) -> np.ndarray | None:
        if self._camera is None:
            return None

        frame_rgb = self._camera.capture_array()
        if frame_rgb is None:
            return None

        if cv2 is None:
            raise ImportError("opencv-python is required to convert Pi camera frames for the vision pipeline")

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._resolution = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        return frame_bgr

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
            finally:
                self._camera.close()
                self._camera = None


def _picamera_present() -> bool:
    if Picamera2 is None:
        return False

    try:
        camera_info = Picamera2.global_camera_info()
    except Exception as exc:  # pragma: no cover - platform-dependent detection
        logger.debug("Picamera2 detection failed: %s", exc)
        return False

    return bool(camera_info)


def _open_webcam_or_raise(width: int, height: int, fps: int, device_index: int) -> WebcamBackend:
    backend = WebcamBackend(device_index=device_index, width=width, height=height, fps=fps)
    return backend.open()


def _open_picamera_or_raise(width: int, height: int, fps: int) -> PiCameraBackend:
    backend = PiCameraBackend(width=width, height=height, fps=fps)
    return backend.open()


def get_camera_backend(
    camera_backend: str,
    width: int = 640,
    height: int = 480,
    fps: int = 10,
    webcam_device_index: int = 0,
) -> CameraBackend:
    """Return an initialized camera backend.

    Auto mode prefers the Pi CSI camera when Picamera2 reports one, then falls
    back to a webcam. Explicit modes skip detection and fail loudly when the
    requested backend cannot be created.
    """

    choice = camera_backend.strip().lower()
    if choice not in {"auto", "webcam", "picamera"}:
        raise ValueError("camera_backend must be one of: auto, webcam, picamera")

    if choice == "webcam":
        return _open_webcam_or_raise(width=width, height=height, fps=fps, device_index=webcam_device_index)

    if choice == "picamera":
        if Picamera2 is None:
            raise RuntimeError(
                "camera_backend='picamera' was requested, but picamera2 is not installed. "
                "Install the Pi-only requirements on the Raspberry Pi or switch camera_backend to 'webcam'."
            )
        if not _picamera_present():
            raise RuntimeError(
                "camera_backend='picamera' was requested, but Picamera2 did not report any CSI camera. "
                "Check the ribbon cable, libcamera setup, and that the sensor is detected."
            )
        try:
            return _open_picamera_or_raise(width=width, height=height, fps=fps)
        except Exception as exc:
            raise RuntimeError(
                "camera_backend='picamera' was requested, but Picamera2 failed to open the CSI camera. "
                "Check libcamera permissions, camera wiring, and whether another process is holding the sensor."
            ) from exc

    # Auto mode: prefer the Pi CSI path when it exists, but recover to webcam
    # without drama if the Pi stack is unavailable on the current platform.
    if _picamera_present():
        try:
            return _open_picamera_or_raise(width=width, height=height, fps=fps)
        except Exception as exc:
            logger.debug("Auto camera selection fell back from Picamera2: %s", exc)

    try:
        return _open_webcam_or_raise(width=width, height=height, fps=fps, device_index=webcam_device_index)
    except Exception as webcam_exc:
        if Picamera2 is None:
            raise RuntimeError(
                "Auto camera selection failed because neither a webcam nor Picamera2-backed CSI camera "
                "could be opened. Install opencv-python for webcam support or the Pi-only camera extras on a Raspberry Pi."
            ) from webcam_exc

        raise RuntimeError(
            "Auto camera selection failed to open any camera backend. "
            f"Webcam error: {webcam_exc}"
        ) from webcam_exc