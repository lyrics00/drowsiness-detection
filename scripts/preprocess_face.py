"""
Detect and crop faces from the raw Face dataset (full upper-body shots),
saving them into data/processed/Face/{train,val,test}/{Active,Fatigue}/

Run from project root:
    python scripts/preprocess_face.py

Requires opencv — already installed as a dependency of mediapipe.
Uses the built-in Haar cascade (no extra downloads needed).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split  # pip install scikit-learn

# ---- paths ----
RAW_FACE_DIR  = Path("data/raw/Face")
OUT_DIR       = Path("data/processed/Face")

# 70% train / 15% val / 15% test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remainder

LABEL_MAP = {
    "Active Subjects": "Active",
    "Fatigue Subjects": "Fatigue",
}

# Haar cascade ships with every OpenCV install
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def detect_and_crop_face(img_bgr: np.ndarray,
                         detector: cv2.CascadeClassifier,
                         pad: float = 0.15) -> np.ndarray | None:
    """Return the largest detected face crop (with padding), or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # pick the largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    ih, iw = img_bgr.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x1 = max(0, x - px);  y1 = max(0, y - py)
    x2 = min(iw, x + w + px); y2 = min(ih, y + h + py)
    return img_bgr[y1:y2, x1:x2]


def process_split(img_paths: list[Path], label: str, split: str,
                  detector: cv2.CascadeClassifier) -> tuple[int, int]:
    out_dir = OUT_DIR / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    saved, skipped = 0, 0
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue
        crop = detect_and_crop_face(img, detector)
        if crop is None:
            skipped += 1
            continue
        cv2.imwrite(str(out_dir / p.name), crop)
        saved += 1
    return saved, skipped


def main() -> None:
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {CASCADE_PATH}")

    total_saved = total_skipped = 0

    for raw_folder, label in LABEL_MAP.items():
        src = RAW_FACE_DIR / raw_folder
        imgs = sorted(src.glob("*.jpg")) + sorted(src.glob("*.jpeg"))
        if not imgs:
            print(f"WARNING: no JPEGs found in {src}")
            continue

        print(f"\n{raw_folder}: {len(imgs)} images found")

        # reproducible split
        train_imgs, tmp = train_test_split(imgs, test_size=1 - TRAIN_RATIO, random_state=42)
        val_imgs, test_imgs = train_test_split(
            tmp, test_size=0.5, random_state=42  # split remainder 50/50
        )

        for split_name, split_imgs in [("train", train_imgs),
                                        ("val",   val_imgs),
                                        ("test",  test_imgs)]:
            s, sk = process_split(split_imgs, label, split_name, detector)
            print(f"  {split_name:5s}: {s} saved, {sk} skipped (no face detected)")
            total_saved   += s
            total_skipped += sk

    print(f"\nDone.  Total saved: {total_saved}  |  skipped: {total_skipped}")
    print(f"Output → {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()