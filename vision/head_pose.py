"""Head-pose estimation using MediaPipe landmarks and solvePnP.

The detector does not know anything about camera selection or user-facing UI.
It only turns one frame into yaw/pitch/roll measurements. The math uses a
generic 3D face model and a small set of stable landmark anchors. The points
are approximate, not a custom-trained face mesh; that is enough for relative
head orientation, which is what study mode needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

try:
    from mediapipe.python.solutions.face_mesh import FaceMesh as _FaceMesh
except Exception:  # pragma: no cover - import path varies across mediapipe builds
    _FaceMesh = None


if _FaceMesh is None:  # pragma: no cover - import guard
    FaceMesh = mp.solutions.face_mesh.FaceMesh
else:
    FaceMesh = _FaceMesh


_LANDMARK_INDEXES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Generic 3D face model points in millimeters-ish units. The scale does not
# matter for orientation; solvePnP cares about the relative geometry.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPoseDetection:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    face_landmarks: Any
    image_points: tuple[tuple[int, int], ...]


def _euler_from_rotation_matrix(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a rotation matrix into yaw, pitch, roll in degrees.

    The exact sign convention can vary across libraries. We only need a stable
    convention for relative comparisons against a calibrated baseline, so the
    important property is consistency across the entire runtime.
    """

    r00, r01, r02 = rotation_matrix[0]
    r10, r11, r12 = rotation_matrix[1]
    r20, r21, r22 = rotation_matrix[2]

    sy = float(np.sqrt(r00 * r00 + r10 * r10))
    singular = sy < 1e-6

    if not singular:
        pitch_rad = np.arctan2(r21, r22)
        yaw_rad = np.arctan2(-r20, sy)
        roll_rad = np.arctan2(r10, r00)
    else:
        pitch_rad = np.arctan2(-r12, r11)
        yaw_rad = np.arctan2(-r20, sy)
        roll_rad = 0.0

    return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad)), float(np.degrees(roll_rad))


def _landmark_to_image_point(landmark: Any, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def _collect_image_points(face_landmarks: Any, width: int, height: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        _landmark_to_image_point(face_landmarks.landmark[index], width, height)
        for index in _LANDMARK_INDEXES.values()
    )


def estimate_head_pose(frame_bgr: np.ndarray, face_landmarks: Any) -> HeadPoseDetection | None:
    """Estimate yaw/pitch/roll from one face's landmarks.

    The solvePnP step aligns the 2D face landmarks with the generic 3D face
    model above. That gives us a head orientation relative to the camera.
    """

    height, width = frame_bgr.shape[:2]
    image_points = np.array(_collect_image_points(face_landmarks, width, height), dtype=np.float64)

    focal_length = float(width)
    camera_matrix = np.array(
        [[focal_length, 0.0, width / 2.0], [0.0, focal_length, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion_coefficients = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _translation_vector = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw_deg, pitch_deg, roll_deg = _euler_from_rotation_matrix(rotation_matrix)
    return HeadPoseDetection(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        face_landmarks=face_landmarks,
        image_points=tuple((int(point[0]), int(point[1])) for point in image_points),
    )


class HeadPoseDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._face_mesh = FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> HeadPoseDetection | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        return estimate_head_pose(frame_bgr, face_landmarks)

    def close(self) -> None:
        self._face_mesh.close()