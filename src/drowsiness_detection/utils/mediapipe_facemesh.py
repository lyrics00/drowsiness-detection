from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    mp = None
    _MP_IMPORT_ERR = e


# Landmark indices for MediaPipe FaceMesh (468 landmarks).
# Commonly used for EAR with MediaPipe; these are stable across many examples.
LEFT_EYE_6 = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_6 = [362, 385, 387, 263, 373, 380]

# Simple mouth points for MAR (left, right, upper, lower)
MOUTH_4 = [61, 291, 13, 14]

# Keypoints for coarse head pose (2D-3D solvePnP)
POSE_2D_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}


@dataclass(frozen=True)
class FaceMeshResult:
    landmarks_xy: np.ndarray  # (468,2) pixel coords
    landmarks_xyz: Optional[np.ndarray]  # (468,3) normalized coords if available

    left_eye_xy: np.ndarray  # (6,2)
    right_eye_xy: np.ndarray  # (6,2)
    mouth_xy: np.ndarray  # (4,2)

    pose_points_2d: np.ndarray  # (6,2)


class FaceMeshDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None:  # pragma: no cover
            raise ImportError(
                "mediapipe is not available. "
                "Install with `pip install mediapipe` and use Python 3.10-3.12.\n"
                f"Original import error: {_MP_IMPORT_ERR}"
            )
        self._mp = mp
        self._fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, bgr: np.ndarray) -> Optional[FaceMeshResult]:
        import cv2

        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = self._fm.process(rgb)
        if not out.multi_face_landmarks:
            return None

        face = out.multi_face_landmarks[0]
        lm = face.landmark

        xy = np.zeros((len(lm), 2), dtype=np.float32)
        xyz = np.zeros((len(lm), 3), dtype=np.float32)
        for i, p in enumerate(lm):
            xy[i, 0] = p.x * w
            xy[i, 1] = p.y * h
            xyz[i, 0] = p.x
            xyz[i, 1] = p.y
            xyz[i, 2] = p.z

        left_eye = xy[np.array(LEFT_EYE_6)]
        right_eye = xy[np.array(RIGHT_EYE_6)]
        mouth = xy[np.array(MOUTH_4)]

        pose_2d = xy[np.array(list(POSE_2D_IDX.values()))]

        return FaceMeshResult(
            landmarks_xy=xy,
            landmarks_xyz=xyz,
            left_eye_xy=left_eye,
            right_eye_xy=right_eye,
            mouth_xy=mouth,
            pose_points_2d=pose_2d,
        )

    def close(self) -> None:
        self._fm.close()

