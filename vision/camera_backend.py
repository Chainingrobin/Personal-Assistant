"""Camera backend abstraction for study mode.

The rest of the vision pipeline talks only to this module. On the Raspberry Pi
5, the CSI camera path now uses the `rpicam-vid` CLI and pipes raw frames over
stdout instead of importing Picamera2/libcamera Python bindings. That avoids an
ABI mismatch in this repo's pyenv-managed Python 3.11.9 venv: the distro
`python3-libcamera` bindings are built for the system Python ABI, not the venv,
so `import libcamera` fails even though the camera hardware itself is present.
"""

from __future__ import annotations
import logging
import os
import signal
import shutil
import subprocess
import getpass
from abc import ABC, abstractmethod
import time
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
        self._capture: Optional["cv2.VideoCapture"] = None

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
    """Capture Pi CSI camera frames through `rpicam-vid` and raw YUV420 stdout.

    The subprocess approach keeps the existing pyenv 3.11.9 venv intact and does
    not depend on the system's libcamera Python extension modules. `rpicam-vid`
    is responsible for sensor access; this class only reads its raw I420 stream
    and converts it to OpenCV BGR frames.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 10):
        super().__init__(width=width, height=height, fps=fps)
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._frame_size: int = 0

    def open(self) -> "PiCameraBackend":
        if cv2 is None:
            raise ImportError("opencv-python is required to convert Pi camera frames") from _CV2_ERROR
        if shutil.which("rpicam-vid") is None:
            raise RuntimeError("rpicam-vid was not found on PATH")
        if self.width % 2 or self.height % 2:
            raise ValueError("Pi camera YUV420 capture requires even width and height")
        command = [
            "rpicam-vid",
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
        # Best-effort: kill any stale rpicam-vid processes owned by this user
        # that could be holding the camera pipeline.
        try:
            user = getpass.getuser()
            subprocess.run(["pkill", "-u", user, "-f", "rpicam-vid"], check=False)
        except Exception:
            pass

        self._frame_size = self.width * self.height * 3 // 2
        # Start rpicam-vid in its own process group so we can reliably kill
        # the whole group (libcamera children) on close.
        self._proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            preexec_fn=os.setsid,
        )

        # libcamera mode negotiation takes real wall-clock time; wait a
        # moment before deciding it failed.
        time.sleep(0.5)

        if self._proc.poll() is not None:
            exit_code = self._proc.returncode
            stderr_output = self._read_stderr()
            self._cleanup_process()
            raise RuntimeError(
                f"rpicam-vid exited (code={exit_code}) before the camera stream started. "
                f"stderr: {stderr_output or '<no stderr output>'}"
            )
        self._resolution = (self.width, self.height)
        self._fps = float(self.requested_fps)
        return self

    def _read_stderr(self) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            stderr_bytes = self._proc.stderr.read()
        except Exception:
            return ""
        return stderr_bytes.decode("utf-8", errors="replace").strip()

    def _cleanup_process(self) -> None:
        if self._proc is None:
            return

        if self._proc.stdout is not None:
            self._proc.stdout.close()
        if self._proc.stderr is not None:
            self._proc.stderr.close()

        self._proc = None

    def _read_exact(self, size: int) -> bytes | None:
        if self._proc is None or self._proc.stdout is None:
            return None
        buffer = bytearray()
        while len(buffer) < size:
            chunk = self._proc.stdout.read(size - len(buffer))
            if not chunk:
                # Pipe closed / EOF. Surface WHY instead of returning None blind.
                exit_code = self._proc.poll()
                stderr_output = self._read_stderr()
                print(
                    f"[camera] rpicam-vid stdout closed mid-read "
                    f"(got {len(buffer)}/{size} bytes, exit_code={exit_code}). "
                    f"stderr: {stderr_output or '<no stderr output>'}"
                )
                return None
            buffer.extend(chunk)
        return bytes(buffer)
    def read_frame(self) -> np.ndarray | None:
        if self._proc is None:
            return None

        raw_frame = self._read_exact(self._frame_size)
        if raw_frame is None:
            return None

        # `yuv420` from rpicam-vid is planar I420: full Y plane, then U and V
        # planes at quarter resolution. OpenCV expects that exact byte layout
        # for COLOR_YUV2BGR_I420, so we reshape before converting to BGR.
        yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
        frame_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        self._resolution = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        return frame_bgr

    def close(self) -> None:
        if self._proc is None:
            return

        try:
            # Kill the whole process group started by preexec_fn=os.setsid
            try:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=2)
            except Exception:
                pass
        except ProcessLookupError:
            pass
        finally:
            self._cleanup_process()


def _rpicam_vid_path() -> str | None:
    return shutil.which("rpicam-vid")


def _rpicam_vid_list_cameras_output(timeout: float = 5.0) -> str:
    command = [_rpicam_vid_path() or "rpicam-vid", "--list-cameras"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    output = f"{result.stdout}\n{result.stderr}".strip()
    return output

def _rpicam_vid_camera_present(timeout: float = 5.0) -> bool:
    binary_path = _rpicam_vid_path()
    if binary_path is None:
        print("[camera] rpicam-vid not found on PATH during auto-detection")
        return False
    try:
        output = _rpicam_vid_list_cameras_output(timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[camera] rpicam-vid --list-cameras failed: {exc!r}")
        return False
    print(f"[camera] rpicam-vid --list-cameras raw output:\n{output}")
    detected = any(
        line.lstrip().startswith(tuple(str(index) for index in range(10))) and ":" in line
        for line in output.splitlines()
    )
    print(f"[camera] camera_present parsed as: {detected}")
    return detected


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

    `auto` and `picamera` use `rpicam-vid` detection instead of Picamera2.
    `webcam` remains unchanged and keeps using OpenCV's VideoCapture path.
    """

    choice = camera_backend.strip().lower()
    print(f"[camera] camera_backend config resolved to: '{choice}'")
    if choice not in {"auto", "webcam", "picamera"}:
        raise ValueError("camera_backend must be one of: auto, webcam, picamera")

    if choice == "webcam":
        return _open_webcam_or_raise(width=width, height=height, fps=fps, device_index=webcam_device_index)

    rpicam_path = _rpicam_vid_path()
    camera_present = _rpicam_vid_camera_present() if rpicam_path is not None else False

    if choice == "picamera":
        if rpicam_path is None:
            raise RuntimeError("camera_backend='picamera' was requested, but rpicam-vid is not installed or not on PATH.")
        if not camera_present:
            raise RuntimeError(
                "camera_backend='picamera' was requested, but rpicam-vid did not report a detected camera. "
                "Check the sensor cable, permissions, and that rpicam-vid --list-cameras shows the Pi camera."
            )
        try:
            return _open_picamera_or_raise(width=width, height=height, fps=fps)
        except Exception as exc:
            raise RuntimeError(
                f"camera_backend='picamera' failed to start rpicam-vid: {exc}"
            ) from exc

    if camera_present:
        try:
            return _open_picamera_or_raise(width=width, height=height, fps=fps)
        except Exception as exc:
            logger.debug("Auto camera selection fell back from rpicam-vid: %s", exc)

    try:
        return _open_webcam_or_raise(width=width, height=height, fps=fps, device_index=webcam_device_index)
    except Exception as webcam_exc:
        if rpicam_path is None:
            raise RuntimeError(
                "Auto camera selection failed because neither rpicam-vid nor a webcam backend could be opened. "
                "Install opencv-python for webcam support or install rpicam-vid on the Raspberry Pi."
            ) from webcam_exc

        raise RuntimeError(
            "Auto camera selection failed to open any camera backend. "
            f"Webcam error: {webcam_exc}"
        ) from webcam_exc