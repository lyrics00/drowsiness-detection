from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _to_xy(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2) points, got {pts.shape}")
    return pts


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(eye_xy: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 landmarks:
      [p1 (outer), p2 (upper1), p3 (upper2), p4 (inner), p5 (lower1), p6 (lower2)]
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    eye = _to_xy(eye_xy)
    if eye.shape[0] != 6:
        raise ValueError("EAR expects exactly 6 eye landmarks")
    p1, p2, p3, p4, p5, p6 = eye
    denom = 2.0 * euclidean(p1, p4)
    if denom < 1e-6:
        return 0.0
    return (euclidean(p2, p6) + euclidean(p3, p5)) / denom


def eye_line_flatness(eye_xy: np.ndarray) -> float:
    """
    Measure how closely the eyelid landmarks follow the corner-to-corner baseline.

    Returns a unitless ratio:
      0.0   -> perfectly flat / closed line
      larger values -> more vertical separation / more open eye
    """
    eye = _to_xy(eye_xy)
    if eye.shape[0] != 6:
        raise ValueError("eye_line_flatness expects exactly 6 eye landmarks")

    p1, _, _, p4, _, _ = eye
    width = euclidean(p1, p4)
    if width < 1e-6:
        return 0.0

    interior = eye[[1, 2, 4, 5]]
    baseline = p4 - p1

    # 2D perpendicular distance from each eyelid point to the eye baseline.
    cross = (
        (interior[:, 0] - p1[0]) * baseline[1]
        - (interior[:, 1] - p1[1]) * baseline[0]
    )
    dists = np.abs(cross) / width
    return float(dists.mean() / width)


def mouth_aspect_ratio(mouth_xy: np.ndarray) -> float:
    """
    Simple MAR from 4 landmarks: [left, right, upper, lower]
    MAR = ||upper-lower|| / ||left-right||
    """
    mouth = _to_xy(mouth_xy)
    if mouth.shape[0] != 4:
        raise ValueError("MAR expects exactly 4 mouth landmarks")
    left, right, upper, lower = mouth
    denom = euclidean(left, right)
    if denom < 1e-6:
        return 0.0
    return euclidean(upper, lower) / denom


def mean_angle_degrees(angles_deg: Iterable[float]) -> float:
    a = np.array(list(angles_deg), dtype=np.float32)
    if a.size == 0:
        return 0.0
    return float(a.mean())


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def bbox_from_landmarks(
    xy: np.ndarray, img_w: int, img_h: int, pad: float = 0.25
) -> Tuple[int, int, int, int]:
    xy = _to_xy(xy)
    x0 = float(xy[:, 0].min())
    y0 = float(xy[:, 1].min())
    x1 = float(xy[:, 0].max())
    y1 = float(xy[:, 1].max())
    w = x1 - x0
    h = y1 - y0
    x0 -= pad * w
    y0 -= pad * h
    x1 += pad * w
    y1 += pad * h
    x0i = int(max(0, min(img_w - 1, round(x0))))
    y0i = int(max(0, min(img_h - 1, round(y0))))
    x1i = int(max(0, min(img_w - 1, round(x1))))
    y1i = int(max(0, min(img_h - 1, round(y1))))
    if x1i <= x0i:
        x1i = min(img_w - 1, x0i + 1)
    if y1i <= y0i:
        y1i = min(img_h - 1, y0i + 1)
    return x0i, y0i, x1i, y1i
