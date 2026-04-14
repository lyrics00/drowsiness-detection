from __future__ import annotations

import argparse
from pathlib import Path

from extract_features_video import extract_video_features


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def iter_videos(root: Path) -> list[tuple[Path, str]]:
    videos: list[tuple[Path, str]] = []
    for cls_name in ["alert", "drowsy"]:
        cls_dir = root / cls_name
        if not cls_dir.exists():
            continue
        for path in sorted(cls_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                videos.append((path, cls_name))
    return videos


def main() -> int:
    p = argparse.ArgumentParser(
        description="Batch-extract EAR/MAR/head-pose CSVs from labeled alert/drowsy video folders."
    )
    p.add_argument("--video_root", type=str, required=True, help="Folder with alert/ and drowsy/ video subfolders.")
    p.add_argument("--out_root", type=str, default="data/processed/features")
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--drop_missing", action="store_true")
    args = p.parse_args()

    video_root = Path(args.video_root)
    out_root = Path(args.out_root)
    videos = iter_videos(video_root)
    if not videos:
        raise FileNotFoundError(
            f"No videos found under {video_root}/alert or {video_root}/drowsy. "
            f"Supported extensions: {sorted(VIDEO_EXTS)}"
        )

    print(f"Found {len(videos)} videos.")
    for idx, (video_path, cls_name) in enumerate(videos, start=1):
        rel = video_path.relative_to(video_root / cls_name)
        out_csv = (out_root / cls_name / rel).with_suffix(".csv")
        print(f"[{idx}/{len(videos)}] {video_path} -> {out_csv}")
        rows = extract_video_features(
            video_path=video_path,
            out_csv=out_csv,
            max_frames=args.max_frames,
            sample_every=max(1, args.sample_every),
            keep_missing=not args.drop_missing,
        )
        print(f"  rows={rows}")

    print(f"Done. CSVs written under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
