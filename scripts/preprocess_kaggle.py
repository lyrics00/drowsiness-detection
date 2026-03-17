from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_tree_images(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(src)
            out = dst / rel.name
            shutil.copy2(p, out)
            n += 1
    return n


def main() -> int:
    """
    Minimal helper to standardize assorted Kaggle datasets into:

      data/processed/train/alert
      data/processed/train/drowsy
      data/processed/val/alert
      data/processed/val/drowsy

    This script is intentionally conservative and may need edits depending on
    the exact Kaggle dataset folder names.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True, help="Path to extracted Kaggle dataset root (data/raw/...).")
    p.add_argument("--dst", type=str, default="data/processed", help="Output root.")
    p.add_argument("--alert_glob", type=str, default="**/*alert*", help="Glob for alert/non-drowsy images.")
    p.add_argument("--drowsy_glob", type=str, default="**/*drowsy*", help="Glob for drowsy images.")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Heuristic collection: pick directories that match globs and copy images.
    alert_dirs = sorted({p for p in src.glob(args.alert_glob) if p.is_dir()})
    drowsy_dirs = sorted({p for p in src.glob(args.drowsy_glob) if p.is_dir()})

    if not alert_dirs or not drowsy_dirs:
        raise SystemExit(
            "Could not find class folders automatically.\n"
            f"alert_dirs={alert_dirs}\n"
            f"drowsy_dirs={drowsy_dirs}\n"
            "Pass different --alert_glob/--drowsy_glob or edit this script to match your dataset."
        )

    # For starter code: copy into train split; later you can do proper stratified split.
    train_alert = dst / "train" / "alert"
    train_drowsy = dst / "train" / "drowsy"
    n_a = sum(copy_tree_images(d, train_alert) for d in alert_dirs)
    n_d = sum(copy_tree_images(d, train_drowsy) for d in drowsy_dirs)

    print(f"Copied alert images:  {n_a}")
    print(f"Copied drowsy images: {n_d}")
    print(f"Output: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

