from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from drowsiness_detection.head_pose import estimate_head_pose_degrees
from drowsiness_detection.utils.geometry import eye_aspect_ratio, mouth_aspect_ratio
from drowsiness_detection.utils.mediapipe_facemesh import FaceMeshDetector


def main() -> int:
    """
    Extract frame-wise handcrafted features (EAR/MAR/head pose) from a video.
    Output CSV can be grouped into sequences later for temporal models.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--max_frames", type=int, default=-1)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    detector = FaceMeshDetector(max_num_faces=1)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "ear", "mar", "yaw_deg", "pitch_deg", "roll_deg"])

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and i >= args.max_frames:
                break

            res = detector.detect(frame)
            if res is None:
                i += 1
                continue

            ear = float((eye_aspect_ratio(res.left_eye_xy) + eye_aspect_ratio(res.right_eye_xy)) / 2.0)
            mar = float(mouth_aspect_ratio(res.mouth_xy))
            h, ww = frame.shape[:2]
            pose = estimate_head_pose_degrees((ww, h), res.pose_points_2d)
            if pose is None:
                yaw = pitch = roll = 0.0
            else:
                yaw, pitch, roll = pose.yaw_deg, pose.pitch_deg, pose.roll_deg

            w.writerow([i, ear, mar, yaw, pitch, roll])
            i += 1

    cap.release()
    detector.close()
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

