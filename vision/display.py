"""Debug display helpers for the study-mode monitor."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import mediapipe as mp

from .head_pose import HeadPoseDetection


_DRAWING_UTILS = mp.solutions.drawing_utils
_FACE_MESH_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION
_LANDMARK_STYLE = _DRAWING_UTILS.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
_CONNECTION_STYLE = _DRAWING_UTILS.DrawingSpec(color=(0, 180, 255), thickness=1, circle_radius=1)
_GUI_DISABLED_REASON: str | None = None
_HEADLESS_PREVIEW_PATH = Path(".aegis_debug/study_mode_latest.jpg")
_LAST_HEADLESS_PREVIEW_LOG_TS: float | None = None
_QT_PLATFORM_FORCED_XCB = False


def _qt_xcb_plugin_available() -> bool:
    plugin_path = Path(cv2.__file__).resolve().parent / "qt" / "plugins" / "platforms" / "libqxcb.so"
    return plugin_path.exists()


def _prefer_xcb_qt_platform() -> None:
    global _QT_PLATFORM_FORCED_XCB
    if _QT_PLATFORM_FORCED_XCB:
        return
    if os.environ.get("QT_QPA_PLATFORM") in {None, "", "wayland"} and _qt_xcb_plugin_available():
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        _QT_PLATFORM_FORCED_XCB = True


_prefer_xcb_qt_platform()


def get_headless_preview_path() -> Path:
    return _HEADLESS_PREVIEW_PATH


def _banner_color(status: str) -> tuple[int, int, int]:
    normalized = status.upper()
    if normalized.startswith("CALIBRATING"):
        return (0, 180, 255)
    if normalized.startswith("LOOKING AWAY"):
        return (0, 0, 255)
    return (0, 200, 0)


def render_debug_frame(
    frame_bgr,
    detection: HeadPoseDetection | None,
    status: str,
    calibration_note: str = "",
):
    output = frame_bgr.copy()

    if detection is not None and detection.face_landmarks is not None:
        _DRAWING_UTILS.draw_landmarks(
            output,
            detection.face_landmarks,
            _FACE_MESH_CONNECTIONS,
            landmark_drawing_spec=_LANDMARK_STYLE,
            connection_drawing_spec=_CONNECTION_STYLE,
        )

    banner_height = 42
    color = _banner_color(status)
    cv2.rectangle(output, (0, 0), (output.shape[1], banner_height), color, thickness=-1)
    cv2.putText(
        output,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if detection is not None:
        cv2.putText(
            output,
            f"yaw: {detection.yaw_deg:6.2f}  pitch: {detection.pitch_deg:6.2f}  roll: {detection.roll_deg:6.2f}",
            (12, output.shape[0] - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if calibration_note:
        cv2.putText(
            output,
            calibration_note,
            (12, output.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def show_debug_frame(window_name: str, frame_bgr) -> None:
    global _GUI_DISABLED_REASON
    if _GUI_DISABLED_REASON is not None:
        _HEADLESS_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(_HEADLESS_PREVIEW_PATH), frame_bgr)
        global _LAST_HEADLESS_PREVIEW_LOG_TS
        from time import monotonic

        now = monotonic()
        if _LAST_HEADLESS_PREVIEW_LOG_TS is None or now - _LAST_HEADLESS_PREVIEW_LOG_TS >= 2.0:
            _LAST_HEADLESS_PREVIEW_LOG_TS = now
            print(f"[study] Debug preview saved to {_HEADLESS_PREVIEW_PATH.resolve()}")
        return

    try:
        cv2.imshow(window_name, frame_bgr)
        cv2.waitKey(1)
    except cv2.error as exc:
        if not _QT_PLATFORM_FORCED_XCB and _qt_xcb_plugin_available():
            _prefer_xcb_qt_platform()
            try:
                cv2.imshow(window_name, frame_bgr)
                cv2.waitKey(1)
                return
            except cv2.error:
                pass

        _GUI_DISABLED_REASON = str(exc)
        print(
            "[study] Debug window unavailable; continuing headless. "
            "Set AEGIS_SHOW_DEBUG_WINDOW=false to silence this message."
        )


def close_debug_window(window_name: str) -> None:
    if _GUI_DISABLED_REASON is not None:
        return
    try:
        cv2.destroyWindow(window_name)
    except Exception:
        cv2.destroyAllWindows()