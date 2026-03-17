from __future__ import annotations

import argparse
from dataclasses import asdict

import cv2
import numpy as np

from drowsiness_detection.config import DemoConfig
from drowsiness_detection.head_pose import estimate_head_pose_degrees
from drowsiness_detection.temporal import TemporalCueTracker
from drowsiness_detection.utils.geometry import eye_aspect_ratio, mouth_aspect_ratio
from drowsiness_detection.utils.mediapipe_facemesh import FaceMeshDetector


def draw_text(img: np.ndarray, lines: list[str], x: int = 10, y: int = 25) -> None:
    for i, line in enumerate(lines):
        yy = y + i * 22
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 230, 20), 2, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser(description="Real-time drowsiness detection demo (MediaPipe + EAR/MAR + temporal cues).")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--ear_closed", type=float, default=None, help="EAR threshold for eye-closed (override config).")
    p.add_argument("--mar_yawn", type=float, default=None, help="MAR threshold for yawn (override config).")
    p.add_argument("--fps", type=float, default=None, help="Assumed FPS for temporal window (override config).")
    args = p.parse_args()

    cfg = DemoConfig(camera_index=args.camera)
    ear_closed = cfg.ear_closed_thresh if args.ear_closed is None else float(args.ear_closed)
    mar_yawn = cfg.mar_yawn_thresh if args.mar_yawn is None else float(args.mar_yawn)
    fps = cfg.fps_assumed if args.fps is None else float(args.fps)

    detector = FaceMeshDetector(max_num_faces=cfg.max_num_faces)
    tracker = TemporalCueTracker(
        fps_assumed=fps,
        window_seconds=cfg.window_seconds,
        ear_closed_thresh=ear_closed,
        mar_yawn_thresh=mar_yawn,
    )

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cfg.camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = detector.detect(frame)
            if res is None:
                draw_text(frame, ["No face detected"])
                cv2.imshow("drowsiness-demo", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            ear_l = eye_aspect_ratio(res.left_eye_xy)
            ear_r = eye_aspect_ratio(res.right_eye_xy)
            ear = float((ear_l + ear_r) / 2.0)
            mar = mouth_aspect_ratio(res.mouth_xy)
            state = tracker.update(ear=ear, mar=mar)

            h, w = frame.shape[:2]
            pose = estimate_head_pose_degrees((w, h), res.pose_points_2d)

            is_drowsy = (state.perclos >= cfg.perclos_drowsy_thresh) or (state.yawn_seconds >= cfg.yawn_min_seconds)

            # Draw landmarks for eyes + mouth
            for pt in np.vstack([res.left_eye_xy, res.right_eye_xy, res.mouth_xy]).astype(np.int32):
                cv2.circle(frame, tuple(pt.tolist()), 2, (0, 255, 255), -1)

            status = "DROWSY" if is_drowsy else "ALERT"
            color = (20, 20, 240) if is_drowsy else (20, 230, 20)
            cv2.putText(frame, status, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

            lines = [
                f"EAR: {state.ear:.3f} (closed<{ear_closed:.2f})  eye_closed={state.is_eye_closed}",
                f"MAR: {state.mar:.3f} (yawn>{mar_yawn:.2f})  yawn_s={state.yawn_seconds:.2f}",
                f"PERCLOS(~{cfg.window_seconds:.0f}s): {state.perclos:.2f} (drowsy>={cfg.perclos_drowsy_thresh:.2f})",
                f"Blink rate (rough): {state.blink_rate_per_min:.1f}/min",
            ]
            if pose is not None:
                lines.append(f"Pose yaw/pitch/roll: {pose.yaw_deg:+.1f} {pose.pitch_deg:+.1f} {pose.roll_deg:+.1f}")
            draw_text(frame, lines)

            cv2.imshow("drowsiness-demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.reset_blink_count()
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

