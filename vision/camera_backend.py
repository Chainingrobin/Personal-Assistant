"""Camera backend abstraction for study mode.

Pi CSI cameras are accessed through rpicam-vid rather than Picamera2/libcamera
Python bindings. This keeps the project compatible with the pyenv-managed
Python environment while allowing the Raspberry Pi's system rpicam/libcamera
stack to handle the CSI camera.

The Pi backend:
    rpicam-vid -> YUV420/I420 stdout -> NumPy -> OpenCV BGR

The backend owns exactly one rpicam-vid process and only terminates that
process. It does NOT globally kill other rpicam-vid processes.
"""

from __future__ import annotations

import logging
import os
import signal
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _CV2_ERROR = exc
else:
    _CV2_ERROR = None


logger = logging.getLogger(__name__)


class CameraBackend(ABC):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("Camera width and height must be positive")

        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(
                "Camera width and height must be even for YUV420 capture"
            )

        if fps <= 0:
            raise ValueError("Camera FPS must be positive")

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
    """OpenCV webcam backend."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
    ):
        super().__init__(width=width, height=height, fps=fps)
        self.device_index = device_index
        self._capture: Optional["cv2.VideoCapture"] = None

    def open(self) -> "WebcamBackend":
        if cv2 is None:
            raise ImportError(
                "opencv-python is required for webcam camera access"
            ) from _CV2_ERROR

        if self._capture is not None:
            return self

        capture = cv2.VideoCapture(self.device_index)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open webcam device index {self.device_index}. "
                "Check permissions and make sure another application is not "
                "using the webcam."
            )

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            float(self.width),
        )
        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            float(self.height),
        )
        capture.set(
            cv2.CAP_PROP_FPS,
            float(self.requested_fps),
        )

        self._capture = capture

        actual_width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        actual_height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        actual_fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        self._resolution = (
            actual_width or self.width,
            actual_height or self.height,
        )

        self._fps = actual_fps or float(self.requested_fps)

        return self

    def read_frame(self) -> np.ndarray | None:
        if self._capture is None:
            return None

        success, frame = self._capture.read()

        if not success or frame is None:
            return None

        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning(
                "Webcam returned unexpected frame shape: %s",
                frame.shape,
            )
            return None

        self._resolution = (
            int(frame.shape[1]),
            int(frame.shape[0]),
        )

        return frame

    def close(self) -> None:
        capture = self._capture
        self._capture = None

        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.exception("Error while closing webcam")


class PiCameraBackend(CameraBackend):
    """Raspberry Pi CSI camera backend using rpicam-vid.

    Frames are captured as YUV420/I420 and converted to OpenCV BGR.
    """

    STARTUP_TIMEOUT = 5.0
    TERMINATE_TIMEOUT = 2.0

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
    ):
        super().__init__(width=width, height=height, fps=fps)

        self._proc: Optional[subprocess.Popen[bytes]] = None

        # I420/YUV420 frame size:
        # Y = width * height
        # U = width/2 * height/2
        # V = width/2 * height/2
        # Total = width * height * 3/2
        self._frame_size = (
            self.width * self.height * 3 // 2
        )

    def open(self) -> "PiCameraBackend":
        if cv2 is None:
            raise ImportError(
                "opencv-python is required for Pi camera frame conversion"
            ) from _CV2_ERROR

        if self._proc is not None:
            # Already open.
            if self._proc.poll() is None:
                return self

            # Process died unexpectedly.
            self._cleanup_process()

        rpicam_path = shutil.which("rpicam-vid")

        if rpicam_path is None:
            raise RuntimeError(
                "rpicam-vid was not found on PATH. "
                "Install the Raspberry Pi camera applications first."
            )

        command = [
            rpicam_path,
            "--nopreview",
            "-t",
            "0",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--framerate",
            str(self.requested_fps),
            "--codec",
            "yuv420",
            "-o",
            "-",
        ]

        logger.info(
            "Starting Pi camera: %s",
            " ".join(command),
        )

        try:
            # stderr goes to DEVNULL deliberately.
            #
            # rpicam/libcamera can continuously emit diagnostic messages.
            # Leaving stderr as PIPE without a reader can eventually fill
            # the pipe and block the camera process.
            #
            # stdout is the actual camera frame stream and remains piped.
            self._proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
            )
        except OSError as exc:
            self._proc = None
            raise RuntimeError(
                f"Failed to start rpicam-vid: {exc}"
            ) from exc

        # Give libcamera/rpicam a short amount of time to initialize.
        deadline = time.monotonic() + self.STARTUP_TIMEOUT

        while time.monotonic() < deadline:
            if self._proc is None:
                break

            exit_code = self._proc.poll()

            if exit_code is not None:
                self._cleanup_process()

                raise RuntimeError(
                    "rpicam-vid exited before the camera stream started "
                    f"(exit code {exit_code})."
                )

            # Don't wait the entire startup timeout if the process is alive.
            # A short delay allows libcamera to finish sensor negotiation.
            time.sleep(0.05)

            # We deliberately don't attempt to read a full frame here.
            # The first call to read_frame() owns that responsibility.

        if self._proc is None or self._proc.poll() is not None:
            self.close()

            raise RuntimeError(
                "rpicam-vid failed to remain running during startup."
            )

        self._resolution = (
            self.width,
            self.height,
        )
        self._fps = float(self.requested_fps)

        logger.info(
            "Pi camera started successfully: %dx%d @ %s FPS",
            self.width,
            self.height,
            self.requested_fps,
        )

        return self

    def _read_exact(self, size: int) -> bytes | None:
        """Read exactly one frame from rpicam-vid stdout.

        Returns None only when the camera process has stopped or stdout
        reaches EOF.
        """

        if self._proc is None:
            return None

        if self._proc.stdout is None:
            return None

        buffer = bytearray()

        while len(buffer) < size:
            # Detect process death before attempting another read.
            exit_code = self._proc.poll()

            if exit_code is not None:
                logger.error(
                    "rpicam-vid exited while reading camera frame "
                    "(exit code %s, received %d/%d bytes)",
                    exit_code,
                    len(buffer),
                    size,
                )
                return None

            try:
                chunk = self._proc.stdout.read(
                    size - len(buffer)
                )
            except (OSError, ValueError) as exc:
                logger.error(
                    "Error reading rpicam-vid stdout: %s",
                    exc,
                )
                return None

            if not chunk:
                logger.error(
                    "rpicam-vid stdout closed while reading frame "
                    "(received %d/%d bytes)",
                    len(buffer),
                    size,
                )
                return None

            buffer.extend(chunk)

        return bytes(buffer)

    def read_frame(self) -> np.ndarray | None:
        if self._proc is None:
            return None

        if self._proc.poll() is not None:
            logger.error(
                "Cannot read camera frame because rpicam-vid "
                "has already exited with code %s",
                self._proc.returncode,
            )
            return None

        raw_frame = self._read_exact(
            self._frame_size
        )

        if raw_frame is None:
            return None

        try:
            yuv_frame = np.frombuffer(
                raw_frame,
                dtype=np.uint8,
            ).reshape(
                (
                    self.height * 3 // 2,
                    self.width,
                )
            )

            frame_bgr = cv2.cvtColor(
                yuv_frame,
                cv2.COLOR_YUV2BGR_I420,
            )

        except Exception as exc:
            logger.exception(
                "Failed to convert Pi camera YUV420 frame: %s",
                exc,
            )
            return None

        if frame_bgr is None:
            return None

        if (
            frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            logger.error(
                "Unexpected converted camera frame shape: %s",
                frame_bgr.shape,
            )
            return None

        self._resolution = (
            int(frame_bgr.shape[1]),
            int(frame_bgr.shape[0]),
        )

        return frame_bgr

    def close(self) -> None:
        """Safely terminate only the rpicam-vid process owned by this backend."""

        proc = self._proc

        if proc is None:
            return

        # Prevent another close() call from operating on the same process.
        self._proc = None

        try:
            if proc.poll() is None:
                try:
                    # start_new_session=True means the process is the leader
                    # of its own process group.
                    os.killpg(
                        os.getpgid(proc.pid),
                        signal.SIGTERM,
                    )
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        proc.terminate()
                    except (ProcessLookupError, OSError):
                        pass

                try:
                    proc.wait(
                        timeout=self.TERMINATE_TIMEOUT
                    )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "rpicam-vid did not terminate cleanly; "
                        "sending SIGKILL"
                    )

                    try:
                        os.killpg(
                            os.getpgid(proc.pid),
                            signal.SIGKILL,
                        )
                    except (
                        ProcessLookupError,
                        PermissionError,
                        OSError,
                    ):
                        try:
                            proc.kill()
                        except (
                            ProcessLookupError,
                            OSError,
                        ):
                            pass

                    try:
                        proc.wait(timeout=1)
                    except (
                        subprocess.TimeoutExpired,
                        ProcessLookupError,
                    ):
                        pass

        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass

            try:
                if proc.stderr is not None:
                    proc.stderr.close()
            except Exception:
                pass

    def _cleanup_process(self) -> None:
        """Internal cleanup for a process that is already dead."""

        proc = self._proc
        self._proc = None

        if proc is None:
            return

        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass

        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass


def _rpicam_vid_path() -> str | None:
    return shutil.which("rpicam-vid")


def _rpicam_vid_list_cameras_output(
    timeout: float = 5.0,
) -> str:
    binary = _rpicam_vid_path()

    if binary is None:
        return ""

    command = [
        binary,
        "--list-cameras",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (
        subprocess.SubprocessError,
        OSError,
    ) as exc:
        logger.warning(
            "rpicam-vid --list-cameras failed: %r",
            exc,
        )
        return ""

    return (
        f"{result.stdout}\n{result.stderr}"
    ).strip()


def _rpicam_vid_camera_present(
    timeout: float = 5.0,
) -> bool:
    binary = _rpicam_vid_path()

    if binary is None:
        print(
            "[camera] rpicam-vid not found on PATH "
            "during auto-detection"
        )
        return False

    output = _rpicam_vid_list_cameras_output(
        timeout=timeout
    )

    print(
        "[camera] rpicam-vid --list-cameras raw output:\n"
        f"{output}"
    )

    detected = any(
        line.lstrip().startswith(
            tuple(str(i) for i in range(10))
        )
        and ":"
        in line
        for line in output.splitlines()
    )

    print(
        f"[camera] camera_present parsed as: {detected}"
    )

    return detected


def _open_webcam_or_raise(
    width: int,
    height: int,
    fps: int,
    device_index: int,
) -> WebcamBackend:
    backend = WebcamBackend(
        device_index=device_index,
        width=width,
        height=height,
        fps=fps,
    )

    try:
        return backend.open()
    except Exception:
        backend.close()
        raise


def _open_picamera_or_raise(
    width: int,
    height: int,
    fps: int,
) -> PiCameraBackend:
    backend = PiCameraBackend(
        width=width,
        height=height,
        fps=fps,
    )

    try:
        return backend.open()
    except Exception:
        backend.close()
        raise


def get_camera_backend(
    camera_backend: str,
    width: int = 640,
    height: int = 480,
    fps: int = 10,
    webcam_device_index: int = 0,
) -> CameraBackend:
    """Return an initialized camera backend.

    Supported modes:

        auto
            Prefer Pi CSI camera when detected, then fall back to webcam.

        picamera
            Require the Raspberry Pi CSI camera.

        webcam
            Require an OpenCV webcam.
    """

    choice = camera_backend.strip().lower()

    print(
        f"[camera] camera_backend config resolved to: "
        f"'{choice}'"
    )

    if choice not in {
        "auto",
        "webcam",
        "picamera",
    }:
        raise ValueError(
            "camera_backend must be one of: "
            "auto, webcam, picamera"
        )

    if choice == "webcam":
        return _open_webcam_or_raise(
            width=width,
            height=height,
            fps=fps,
            device_index=webcam_device_index,
        )

    rpicam_path = _rpicam_vid_path()

    camera_present = (
        _rpicam_vid_camera_present()
        if rpicam_path is not None
        else False
    )

    if choice == "picamera":
        if rpicam_path is None:
            raise RuntimeError(
                "camera_backend='picamera' was requested, "
                "but rpicam-vid is not installed or not on PATH."
            )

        if not camera_present:
            raise RuntimeError(
                "camera_backend='picamera' was requested, "
                "but rpicam-vid did not report a detected camera. "
                "Check the CSI cable and run "
                "'rpicam-vid --list-cameras'."
            )

        try:
            return _open_picamera_or_raise(
                width=width,
                height=height,
                fps=fps,
            )
        except Exception as exc:
            raise RuntimeError(
                "camera_backend='picamera' failed to start "
                f"rpicam-vid: {exc}"
            ) from exc

    # auto mode
    if camera_present:
        try:
            return _open_picamera_or_raise(
                width=width,
                height=height,
                fps=fps,
            )
        except Exception as exc:
            logger.warning(
                "Pi CSI camera detected but could not be opened: %s",
                exc,
            )

    try:
        return _open_webcam_or_raise(
            width=width,
            height=height,
            fps=fps,
            device_index=webcam_device_index,
        )
    except Exception as webcam_exc:
        if rpicam_path is None:
            raise RuntimeError(
                "Auto camera selection failed: neither "
                "rpicam-vid nor a webcam backend is available."
            ) from webcam_exc

        raise RuntimeError(
            "Auto camera selection failed to open any camera "
            f"backend. Webcam error: {webcam_exc}"
        ) from webcam_exc