from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def estimate_head_pose_degrees(
    image_size_wh: Tuple[int, int],
    points_2d: np.ndarray,
) -> Optional[HeadPose]:
    """
    Coarse head pose via solvePnP.

    points_2d: (6,2) corresponding to:
      nose_tip, chin, left_eye_outer, right_eye_outer, mouth_left, mouth_right
    """
    import cv2

    w, h = int(image_size_wh[0]), int(image_size_wh[1])
    pts2d = np.asarray(points_2d, dtype=np.float32)
    if pts2d.shape != (6, 2):
        return None

    # Approximate 3D model points (mm) for a generic face.
    model_3d = np.array(
        [
            (0.0, 0.0, 0.0),  # nose tip
            (0.0, -63.6, -12.5),  # chin
            (-43.3, 32.7, -26.0),  # left eye outer
            (43.3, 32.7, -26.0),  # right eye outer
            (-28.9, -28.9, -24.1),  # mouth left
            (28.9, -28.9, -24.1),  # mouth right
        ],
        dtype=np.float32,
    )

    focal_length = w
    center = (w / 2.0, h / 2.0)
    cam = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float32,
    )
    dist = np.zeros((4, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(model_3d, pts2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix into Euler angles.
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0.0

    pitch = float(np.degrees(x))
    yaw = float(np.degrees(y))
    roll = float(np.degrees(z))
    return HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)

