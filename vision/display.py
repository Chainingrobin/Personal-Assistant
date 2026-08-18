"""Debug display helpers for the study-mode monitor."""

from __future__ import annotations

import cv2
import mediapipe as mp

from .head_pose import HeadPoseDetection


_DRAWING_UTILS = mp.solutions.drawing_utils
_FACE_MESH_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION
_LANDMARK_STYLE = _DRAWING_UTILS.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
_CONNECTION_STYLE = _DRAWING_UTILS.DrawingSpec(color=(0, 180, 255), thickness=1, circle_radius=1)


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
    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(1)


def close_debug_window(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
    except Exception:
        cv2.destroyAllWindows()