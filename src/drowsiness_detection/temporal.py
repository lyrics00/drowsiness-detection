from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .utils.geometry import clamp01


@dataclass
class TemporalState:
    ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    is_eye_closed: bool = False
    yawn_seconds: float = 0.0
    is_yawning: bool = False

    blink_count_window: int = 0
    blink_rate_per_min: float = 0.0


class TemporalCueTracker:
    """
    Keeps a rolling window of EAR/MAR to compute:
      - Eye-closed ratio (PERCLOS proxy)
      - Blink count/rate
      - Yawn duration (seconds above MAR threshold)
    """

    def __init__(
        self,
        fps_assumed: float,
        window_seconds: float,
        ear_closed_thresh: float,
        mar_yawn_thresh: float,
    ) -> None:
        self.fps = float(fps_assumed)
        self.window_frames = max(1, int(round(self.fps * float(window_seconds))))
        self.ear_closed_thresh = float(ear_closed_thresh)
        self.mar_yawn_thresh = float(mar_yawn_thresh)

        self._ears = deque(maxlen=self.window_frames)
        self._mars = deque(maxlen=self.window_frames)
        self._eye_closed = deque(maxlen=self.window_frames)  # bools

        self._prev_eye_closed = False
        self._blink_count = 0

        self._yawn_frames_current = 0

    def update(self, ear: float, mar: float) -> TemporalState:
        ear = float(ear)
        mar = float(mar)

        eye_closed = ear < self.ear_closed_thresh
        self._ears.append(ear)
        self._mars.append(mar)
        self._eye_closed.append(eye_closed)

        # Blink: falling edge (open->closed) or rising? Use closed->open completion.
        if self._prev_eye_closed and (not eye_closed):
            self._blink_count += 1
        self._prev_eye_closed = eye_closed

        # Yawn duration
        if mar > self.mar_yawn_thresh:
            self._yawn_frames_current += 1
        else:
            self._yawn_frames_current = 0

        perclos = float(np.mean(np.array(self._eye_closed, dtype=np.float32))) if self._eye_closed else 0.0
        perclos = clamp01(perclos)

        minutes = (len(self._eye_closed) / self.fps) / 60.0
        blink_rate = (self._blink_count / minutes) if minutes > 1e-6 else 0.0

        return TemporalState(
            ear=ear,
            mar=mar,
            perclos=perclos,
            is_eye_closed=eye_closed,
            yawn_seconds=self._yawn_frames_current / self.fps,
            is_yawning=(self._yawn_frames_current > 0),
            blink_count_window=self._blink_count,
            blink_rate_per_min=float(blink_rate),
        )

    def reset_blink_count(self) -> None:
        self._blink_count = 0

