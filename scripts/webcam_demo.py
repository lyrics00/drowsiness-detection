from __future__ import annotations

import argparse
from dataclasses import asdict

import cv2 # type: ignore
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms # type: ignore
from PIL import Image

from drowsiness_detection.config import DemoConfig # type: ignore
from drowsiness_detection.head_pose import estimate_head_pose_degrees # type: ignore
from drowsiness_detection.temporal import TemporalCueTracker # type: ignore
from drowsiness_detection.utils.geometry import eye_aspect_ratio, mouth_aspect_ratio # type: ignore
from drowsiness_detection.utils.mediapipe_facemesh import FaceMeshDetector # type: ignore


# Neural Network Architecture (must match training script)
class SimpleEyeCNN(nn.Module):
    def __init__(self):
        super(SimpleEyeCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


def crop_eye(frame: np.ndarray, pts: np.ndarray, padding: int = 15) -> np.ndarray:
    """Crop the bounding box of the eye from the frame."""
    x_min, y_min = np.min(pts, axis=0).astype(int)
    x_max, y_max = np.max(pts, axis=0).astype(int)
    
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_max = min(frame.shape[0], y_max + padding)
    
    crop = frame[y_min:y_max, x_min:x_max]
    return crop


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

    # Initialize PyTorch Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading CNN on device: {device}")
    cnn_model = SimpleEyeCNN().to(device)
    
    try:
        cnn_model.load_state_dict(torch.load("outputs/runs/eye_cnn_model.pth", map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Are you sure you ran scripts/train_cnn.py first?")
        return 1
        
    cnn_model.eval()

    # Define transformations for the cropped eyes
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

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

            # Extract eyes, transform, and predict with CNN
            cnn_prediction = "Unknown"
            left_eye_crop = crop_eye(frame, res.left_eye_xy)
            right_eye_crop = crop_eye(frame, res.right_eye_xy)

            # Check if crops are valid before passing to CNN
            if left_eye_crop.size > 0 and right_eye_crop.size > 0:
                img_l = Image.fromarray(cv2.cvtColor(left_eye_crop, cv2.COLOR_BGR2RGB))
                img_r = Image.fromarray(cv2.cvtColor(right_eye_crop, cv2.COLOR_BGR2RGB))
                
                tensor_l = transform(img_l).unsqueeze(0).to(device)
                tensor_r = transform(img_r).unsqueeze(0).to(device)

                with torch.no_grad():
                    out_l = cnn_model(tensor_l)
                    out_r = cnn_model(tensor_r)
                    pred_l = torch.argmax(out_l, dim=1).item()
                    pred_r = torch.argmax(out_r, dim=1).item()
                
                # ImageFolder sorts classes alphabetically: 0 = Closed, 1 = Opened
                eyes_are_closed = (pred_l == 0) and (pred_r == 0)
                cnn_prediction = "Closed" if eyes_are_closed else "Opened"
                
                # Use CNN output as fake EAR: below threshold if closed, above if opened
                mock_ear = 0.0 if eyes_are_closed else 1.0 
            else:
                mock_ear = float((eye_aspect_ratio(res.left_eye_xy) + eye_aspect_ratio(res.right_eye_xy)) / 2.0)

            mar = mouth_aspect_ratio(res.mouth_xy)
            state = tracker.update(ear=mock_ear, mar=mar)

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
                f"CNN Eye State: {cnn_prediction}",
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