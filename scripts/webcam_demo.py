"""
Real-time drowsiness detection — fused EyeCNN + FaceCNN (MobileNetV2).

Run from project root:
    python scripts/webcam_demo.py
"""

from __future__ import annotations

import argparse

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models  # type: ignore
from PIL import Image

from drowsiness_detection.config import DemoConfig  # type: ignore
from drowsiness_detection.head_pose import estimate_head_pose_degrees  # type: ignore
from drowsiness_detection.temporal import TemporalCueTracker  # type: ignore
from drowsiness_detection.utils.geometry import eye_aspect_ratio, mouth_aspect_ratio  # type: ignore
from drowsiness_detection.utils.mediapipe_facemesh import FaceMeshDetector  # type: ignore


# ===========================================================================
# Model definitions — must stay in sync with training scripts
# ===========================================================================

class SimpleEyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2),
        )
    def forward(self, x): return self.fc_layer(self.conv_layer(x))


class FaceDrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone        = models.mobilenet_v2(weights=None)
        self.features   = backbone.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
    def forward(self, x): return self.classifier(self.features(x))


# ===========================================================================
# Utilities
# ===========================================================================

def load_model(model: nn.Module, path: str, device: torch.device) -> bool:
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"Loaded {path}")
        return True
    except Exception as e:
        print(f"WARNING: could not load {path}: {e}")
        return False


def crop_eye(frame: np.ndarray, pts: np.ndarray, padding: int = 15) -> np.ndarray:
    x_min, y_min = np.min(pts, axis=0).astype(int)
    x_max, y_max = np.max(pts, axis=0).astype(int)
    return frame[
        max(0, y_min - padding) : min(frame.shape[0], y_max + padding),
        max(0, x_min - padding) : min(frame.shape[1], x_max + padding),
    ]


def crop_face_haar(frame: np.ndarray,
                   detector: cv2.CascadeClassifier,
                   pad: float = 0.15) -> np.ndarray | None:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    ih, iw = frame.shape[:2]
    px, py = int(w * pad), int(h * pad)
    return frame[
        max(0, y - py) : min(ih, y + h + py),
        max(0, x - px) : min(iw, x + w + px),
    ]


def get_eye_closed_prob(model: nn.Module, crop: np.ndarray,
                        tf: transforms.Compose, device: torch.device) -> float:
    """Return probability [0,1] that eye is CLOSED (class 0)."""
    img    = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return F.softmax(model(tensor), dim=1)[0, 0].item()


def predict_prob(model: nn.Module, crop: np.ndarray,
                 tf: transforms.Compose, device: torch.device,
                 class_idx: int) -> float:
    img    = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return F.softmax(model(tensor), dim=1)[0, class_idx].item()


def cnn_ear(p_closed: float, ear_closed_thresh: float, geo_ear: float) -> float:
    """
    Convert a CNN closed-probability into an EAR-like value.

    Key idea: instead of a hard threshold that either fully trusts the CNN or
    fully ignores it, we ALWAYS blend the CNN signal with geo_ear.

    - p_closed=1.0 → returns 0.0  (definitely closed)
    - p_closed=0.5 → returns ear_closed_thresh  (right on the boundary)
    - p_closed=0.0 → returns a comfortable open value

    Then we blend with geo_ear so that when the CNN is uncertain (~0.5),
    the geometric EAR still has influence.
    """
    open_val = max(ear_closed_thresh + 0.10, 0.38)

    # Map p_closed linearly: 0 → open_val, 1 → 0
    cnn_signal = open_val * (1.0 - p_closed)

    # Blend weight: CNN gets more weight the more confident it is
    # confidence = how far p_closed is from 0.5
    confidence = abs(p_closed - 0.5) * 2.0   # 0 = totally unsure, 1 = fully confident
    cnn_weight = 0.3 + 0.7 * confidence       # always at least 30% CNN, up to 100%

    blended = cnn_weight * cnn_signal + (1.0 - cnn_weight) * geo_ear
    return float(np.clip(blended, 0.0, 1.0))


def draw_text(img: np.ndarray, lines: list[str], x: int = 10, y: int = 25) -> None:
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 230, 20), 2, cv2.LINE_AA)


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--camera",      type=int,   default=0)
    p.add_argument("--ear_closed",  type=float, default=None)
    p.add_argument("--mar_yawn",    type=float, default=None)
    p.add_argument("--fps",         type=float, default=None)
    p.add_argument("--eye_weight",  type=float, default=0.6)
    p.add_argument("--face_weight", type=float, default=0.4)
    args = p.parse_args()

    cfg        = DemoConfig(camera_index=args.camera)
    ear_closed = cfg.ear_closed_thresh if args.ear_closed is None else float(args.ear_closed)
    mar_yawn   = cfg.mar_yawn_thresh   if args.mar_yawn   is None else float(args.mar_yawn)
    fps        = cfg.fps_assumed       if args.fps        is None else float(args.fps)

    total_w = args.eye_weight + args.face_weight
    eye_w   = args.eye_weight  / total_w
    face_w  = args.face_weight / total_w

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    eye_model  = SimpleEyeCNN().to(device)
    face_model = FaceDrowsinessCNN().to(device)

    eye_ok  = load_model(eye_model,  "outputs/runs/eye_cnn_model.pth",  device)
    face_ok = load_model(face_model, "outputs/runs/face_cnn_model.pth", device)

    if not eye_ok:
        print("Eye model required. Run scripts/train_cnn.py first.")
        return 1
    if not face_ok:
        print("Face model not loaded — running on eye CNN only.")
        face_w, eye_w = 0.0, 1.0

    eye_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    face_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar      = cv2.CascadeClassifier(haar_path)

    detector = FaceMeshDetector(max_num_faces=cfg.max_num_faces)
    tracker  = TemporalCueTracker(
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

            # ---- geometric EAR (always computed) ----
            geo_ear = float(
                (eye_aspect_ratio(res.left_eye_xy) + eye_aspect_ratio(res.right_eye_xy)) / 2.0
            )

            # ---- EyeCNN: soft blend of CNN prob + geo_ear ----
            eye_signal = geo_ear
            eye_label  = "eye:geo"
            l_crop = crop_eye(frame, res.left_eye_xy)
            r_crop = crop_eye(frame, res.right_eye_xy)
            if l_crop.size > 0 and r_crop.size > 0:
                p_closed_l = get_eye_closed_prob(eye_model, l_crop, eye_tf, device)
                p_closed_r = get_eye_closed_prob(eye_model, r_crop, eye_tf, device)
                p_closed   = (p_closed_l + p_closed_r) / 2.0

                eye_signal = cnn_ear(p_closed, ear_closed, geo_ear)

                state_str  = "Closed" if p_closed > 0.5 else "Open"
                eye_label  = f"eye:{state_str}(cnn={p_closed:.2f})"

            # ---- FaceCNN ----
            face_signal = 0.5
            face_label  = "face:n/a"
            if face_ok and face_w > 0:
                face_crop = crop_face_haar(frame, haar)
                if face_crop is not None and face_crop.size > 0:
                    p_fatigue   = predict_prob(face_model, face_crop, face_tf, device, 1)
                    open_val    = max(ear_closed + 0.05, 0.35)
                    face_signal = float(np.clip(open_val * (1.0 - p_fatigue), 0.0, open_val))
                    face_label  = (f"face:{'Fatigue' if p_fatigue > 0.5 else 'Active'}"
                                   f"({p_fatigue:.2f})")

            # ---- weighted fusion ----
            if face_ok and face_w > 0:
                fused_ear = eye_w * eye_signal + face_w * face_signal
            else:
                fused_ear = eye_signal
            fused_ear = float(np.clip(fused_ear, 0.0, 1.0))

            mar   = mouth_aspect_ratio(res.mouth_xy)
            state = tracker.update(ear=fused_ear, mar=mar)

            h, w  = frame.shape[:2]
            pose  = estimate_head_pose_degrees((w, h), res.pose_points_2d)

            is_drowsy = (
                state.perclos >= cfg.perclos_drowsy_thresh
                or state.yawn_seconds >= cfg.yawn_min_seconds
            )

            for pt in np.vstack([res.left_eye_xy, res.right_eye_xy, res.mouth_xy]).astype(np.int32):
                cv2.circle(frame, tuple(pt.tolist()), 2, (0, 255, 255), -1)

            status = "DROWSY" if is_drowsy else "ALERT"
            cv2.putText(frame, status, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (20, 20, 240) if is_drowsy else (20, 230, 20), 3, cv2.LINE_AA)

            lines = [
                f"{eye_label}   {face_label}",
                f"geo:{geo_ear:.3f}  eye:{eye_signal:.3f}  "
                f"face:{face_signal:.3f}  fused:{fused_ear:.3f}",
                f"eye_closed={state.is_eye_closed}  (thresh<{ear_closed:.2f})",
                f"MAR:{state.mar:.3f} (yawn>{mar_yawn:.2f})  yawn_s:{state.yawn_seconds:.2f}",
                f"PERCLOS({cfg.window_seconds:.0f}s):{state.perclos:.2f}"
                f"  (drowsy>={cfg.perclos_drowsy_thresh:.2f})",
                f"Blink:{state.blink_rate_per_min:.1f}/min",
            ]
            if pose is not None:
                lines.append(f"Pose: yaw{pose.yaw_deg:+.1f} "
                             f"pitch{pose.pitch_deg:+.1f} roll{pose.roll_deg:+.1f}")
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