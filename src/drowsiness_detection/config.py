from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoConfig:
    camera_index: int = 0
    max_num_faces: int = 1

    # Temporal windows
    fps_assumed: float = 30.0
    window_seconds: float = 10.0  # for PERCLOS + smoothing

    # Heuristic thresholds (tune per subject/camera)
    ear_closed_thresh: float = 0.21
    mar_yawn_thresh: float = 0.65
    eye_flatness_closed_thresh: float = 0.035
    eye_flatness_open_thresh: float = 0.045

    # Decision thresholds
    perclos_drowsy_thresh: float = 0.40  # % of frames "eyes closed" within window
    yawn_min_seconds: float = 1.2
