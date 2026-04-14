from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from drowsiness_detection.head_pose import estimate_head_pose_degrees
from drowsiness_detection.utils.geometry import eye_aspect_ratio, mouth_aspect_ratio
from drowsiness_detection.utils.mediapipe_facemesh import FaceMeshDetector


CSV_COLUMNS = [
    "frame",
    "timestamp_sec",
    "face_detected",
    "ear",
    "mar",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
]


def extract_video_features(
    video_path: str | Path,
    out_csv: str | Path,
    max_frames: int = -1,
    sample_every: int = 1,
    keep_missing: bool = True,
) -> int:
    video_path = Path(video_path)
    out = Path(out_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    detector = FaceMeshDetector(max_num_faces=1)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and i >= max_frames:
                break
            if sample_every > 1 and i % sample_every != 0:
                i += 1
                continue

            timestamp_sec = (i / fps) if fps > 0 else 0.0
            res = detector.detect(frame)
            if res is None:
                if keep_missing:
                    w.writerow([i, timestamp_sec, 0, np.nan, np.nan, np.nan, np.nan, np.nan])
                    rows_written += 1
                i += 1
                continue

            ear = float((eye_aspect_ratio(res.left_eye_xy) + eye_aspect_ratio(res.right_eye_xy)) / 2.0)
            mar = float(mouth_aspect_ratio(res.mouth_xy))
            h, ww = frame.shape[:2]
            pose = estimate_head_pose_degrees((ww, h), res.pose_points_2d)
            if pose is None:
                yaw = pitch = roll = np.nan
            else:
                yaw, pitch, roll = pose.yaw_deg, pose.pitch_deg, pose.roll_deg

            w.writerow([i, timestamp_sec, 1, ear, mar, yaw, pitch, roll])
            rows_written += 1
            i += 1

    cap.release()
    detector.close()
    return rows_written


def main() -> int:
    """
    Extract frame-wise handcrafted features (EAR/MAR/head pose) from a video.
    Output CSV can be grouped into sequences later for temporal models.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--drop_missing", action="store_true", help="Skip frames where no face is detected.")
    args = p.parse_args()

    rows = extract_video_features(
        video_path=args.video,
        out_csv=args.out_csv,
        max_frames=args.max_frames,
        sample_every=max(1, args.sample_every),
        keep_missing=not args.drop_missing,
    )
    print(f"Wrote: {args.out_csv} ({rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

