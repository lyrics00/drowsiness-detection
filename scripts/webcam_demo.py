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
from drowsiness_detection.utils.geometry import (  # type: ignore
    bbox_from_landmarks,
    eye_aspect_ratio,
    eye_line_flatness,
    mouth_aspect_ratio,
)
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


def update_eye_closed_state(
    prev_closed: bool,
    flatness: float,
    closed_thresh: float,
    open_thresh: float,
) -> bool:
    """Apply simple hysteresis so closure does not flicker near the boundary."""
    if prev_closed:
        return flatness < open_thresh
    return flatness <= closed_thresh


def draw_eye_tracker(
    img: np.ndarray,
    eye_xy: np.ndarray,
    label: str,
    closed: bool,
) -> None:
    h, w = img.shape[:2]
    color = (20, 20, 240) if closed else (20, 230, 20)
    pts = eye_xy.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    x0, y0, x1, _ = bbox_from_landmarks(eye_xy, w, h, pad=0.55)
    text_y = max(18, y0 - 6)
    cv2.putText(
        img,
        label,
        (x0, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--camera",      type=int,   default=0)
    p.add_argument("--ear_closed",  type=float, default=None)
    p.add_argument("--mar_yawn",    type=float, default=None)
    p.add_argument("--eye_flat", "--eye_flat_closed", dest="eye_flat_closed", type=float, default=None)
    p.add_argument("--eye_flat_open", type=float, default=None)
    p.add_argument("--fps",         type=float, default=None)
    p.add_argument("--eye_weight",  type=float, default=0.6)
    p.add_argument("--face_weight", type=float, default=0.4)
    args = p.parse_args()

    cfg        = DemoConfig(camera_index=args.camera)
    ear_closed = cfg.ear_closed_thresh if args.ear_closed is None else float(args.ear_closed)
    mar_yawn   = cfg.mar_yawn_thresh   if args.mar_yawn   is None else float(args.mar_yawn)
    eye_flat_closed = (
        cfg.eye_flatness_closed_thresh
        if args.eye_flat_closed is None
        else float(args.eye_flat_closed)
    )
    eye_flat_open = (
        cfg.eye_flatness_open_thresh
        if args.eye_flat_open is None
        else float(args.eye_flat_open)
    )
    fps        = cfg.fps_assumed       if args.fps        is None else float(args.fps)

    if eye_flat_open <= eye_flat_closed:
        raise ValueError(
            f"--eye_flat_open must be greater than --eye_flat/--eye_flat_closed "
            f"({eye_flat_open:.3f} <= {eye_flat_closed:.3f})"
        )

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

    left_eye_closed_flat = False
    right_eye_closed_flat = False

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
            left_ear_raw = eye_aspect_ratio(res.left_eye_xy)
            right_ear_raw = eye_aspect_ratio(res.right_eye_xy)
            left_flat = eye_line_flatness(res.left_eye_xy)
            right_flat = eye_line_flatness(res.right_eye_xy)
            left_eye_closed_flat = update_eye_closed_state(
                left_eye_closed_flat,
                left_flat,
                eye_flat_closed,
                eye_flat_open,
            )
            right_eye_closed_flat = update_eye_closed_state(
                right_eye_closed_flat,
                right_flat,
                eye_flat_closed,
                eye_flat_open,
            )
            both_eyes_closed_flat = left_eye_closed_flat and right_eye_closed_flat

            left_ear = left_ear_raw
            right_ear = right_ear_raw
            geo_ear = float((left_ear + right_ear) / 2.0)

            left_eye_signal = left_ear
            right_eye_signal = right_ear

            # ---- EyeCNN: soft blend of CNN prob + geo_ear ----
            eye_signal = geo_ear
            eye_label  = "eye:geo"
            l_crop = crop_eye(frame, res.left_eye_xy)
            r_crop = crop_eye(frame, res.right_eye_xy)
            if l_crop.size > 0 and r_crop.size > 0:
                p_closed_l = get_eye_closed_prob(eye_model, l_crop, eye_tf, device)
                p_closed_r = get_eye_closed_prob(eye_model, r_crop, eye_tf, device)
                left_eye_signal = cnn_ear(p_closed_l, ear_closed, left_ear)
                right_eye_signal = cnn_ear(p_closed_r, ear_closed, right_ear)
                p_closed = (p_closed_l + p_closed_r) / 2.0

                eye_signal = float((left_eye_signal + right_eye_signal) / 2.0)

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
            state = tracker.update(
                ear=fused_ear,
                mar=mar,
                eye_closed_override=both_eyes_closed_flat,
                left_eye_closed=left_eye_closed_flat,
                right_eye_closed=right_eye_closed_flat,
            )

            h, w  = frame.shape[:2]
            pose  = estimate_head_pose_degrees((w, h), res.pose_points_2d)

            is_drowsy = (
                state.perclos >= cfg.perclos_drowsy_thresh
                or state.yawn_seconds >= cfg.yawn_min_seconds
            )

            for pt in np.vstack([res.left_eye_xy, res.right_eye_xy, res.mouth_xy]).astype(np.int32):
                cv2.circle(frame, tuple(pt.tolist()), 2, (0, 255, 255), -1)

            draw_eye_tracker(
                frame,
                res.left_eye_xy,
                f"L {'CLOSED' if state.left_eye_closed else 'OPEN'}",
                state.left_eye_closed,
            )
            draw_eye_tracker(
                frame,
                res.right_eye_xy,
                f"R {'CLOSED' if state.right_eye_closed else 'OPEN'}",
                state.right_eye_closed,
            )

            status = "DROWSY" if is_drowsy else "ALERT"
            cv2.putText(frame, status, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (20, 20, 240) if is_drowsy else (20, 230, 20), 3, cv2.LINE_AA)

            lines = [
                f"{eye_label}   {face_label}",
                f"geo:{geo_ear:.3f}  eye:{eye_signal:.3f}  "
                f"face:{face_signal:.3f}  fused:{fused_ear:.3f}",
                f"L eye:{left_eye_signal:.3f} flat:{left_flat:.3f} "
                f"{'CLOSED' if state.left_eye_closed else 'OPEN'} "
                f"t:{state.left_eye_closed_seconds:.2f}s",
                f"R eye:{right_eye_signal:.3f} flat:{right_flat:.3f} "
                f"{'CLOSED' if state.right_eye_closed else 'OPEN'} "
                f"t:{state.right_eye_closed_seconds:.2f}s",
                f"flat hysteresis close<={eye_flat_closed:.3f} "
                f"open>={eye_flat_open:.3f}  "
                f"both_closed={state.is_eye_closed}",
                f"L/R PERCLOS:{state.left_perclos:.2f}/{state.right_perclos:.2f}  "
                f"overall:{state.perclos:.2f}",
                f"EAR diag thresh<{ear_closed:.2f}  "
                f"(signal only, not drowsy gate)",
                f"MAR:{state.mar:.3f} (yawn>{mar_yawn:.2f})  yawn_s:{state.yawn_seconds:.2f}",
                f"PERCLOS driven by flatness ({cfg.window_seconds:.0f}s)"
                f"  drowsy>={cfg.perclos_drowsy_thresh:.2f}",
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
