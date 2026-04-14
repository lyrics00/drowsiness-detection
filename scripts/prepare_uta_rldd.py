from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
LABEL_MAP = {"0": "alert", "10": "drowsy"}


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(src, dst)


def iter_uta_videos(root: Path, length: str) -> list[tuple[Path, str, str]]:
    length_root = root / length
    if not length_root.exists():
        raise FileNotFoundError(f"Missing clip length folder: {length_root}")

    videos: list[tuple[Path, str, str]] = []
    for subject_dir in sorted(p for p in length_root.iterdir() if p.is_dir()):
        subject = subject_dir.name
        for source_label, class_name in LABEL_MAP.items():
            label_dir = subject_dir / source_label
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                    videos.append((path, class_name, subject))
    return videos


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage UTA-RLDD cropped-face videos into alert/drowsy folders without duplicating data by default."
    )
    p.add_argument(
        "--source_root",
        type=str,
        default=(
            "data/external/kagglehub/datasets/mathiasviborg/"
            "uta-rldd-videos-cropped-by-faces/versions/1/UTA-RLDD Face Cropped Video"
        ),
    )
    p.add_argument("--length", choices=["len5", "len10", "len20", "len30", "len60"], default="len10")
    p.add_argument("--out_root", type=str, default="data/raw/videos/uta_rldd_len10")
    p.add_argument("--mode", choices=["hardlink", "copy", "auto"], default="auto")
    args = p.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    videos = iter_uta_videos(source_root, args.length)
    if not videos:
        raise FileNotFoundError(f"No videos found in {source_root / args.length}")

    counts = {"alert": 0, "drowsy": 0}
    for src, class_name, subject in videos:
        dst = out_root / class_name / subject / src.name
        link_or_copy(src, dst, args.mode)
        counts[class_name] += 1

    print(f"Staged {len(videos)} {args.length} videos under {out_root}")
    print(f"alert={counts['alert']} drowsy={counts['drowsy']}")
    print("Labels: UTA-RLDD 0 -> alert, 10 -> drowsy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
