from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
CLASS_TO_LABEL = {"alert": 0, "drowsy": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}


@dataclass(frozen=True)
class VideoSample:
    path: Path
    label: int
    group: str


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyFrameEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaceVideoTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
        num_frames: int = 32,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.frame_encoder = TinyFrameEncoder(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = PositionalEncoding(embed_dim, max_len=num_frames + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        frame_tokens = self.frame_encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, frame_tokens], dim=1)
        tokens = self.pos(tokens)
        encoded = self.temporal_encoder(tokens)
        return self.head(encoded[:, 0])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_samples(root: Path, max_videos_per_class: int | None = None) -> list[VideoSample]:
    samples: list[VideoSample] = []
    for cls_name, label in CLASS_TO_LABEL.items():
        cls_dir = root / cls_name
        class_samples: list[VideoSample] = []
        for path in sorted(cls_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                rel = path.relative_to(cls_dir)
                group = rel.parts[0] if len(rel.parts) > 1 else path.stem
                class_samples.append(VideoSample(path=path, label=label, group=group))
        if max_videos_per_class is not None:
            by_group: dict[str, list[VideoSample]] = {}
            for sample in class_samples:
                by_group.setdefault(sample.group, []).append(sample)
            class_samples = []
            while len(class_samples) < max_videos_per_class and any(by_group.values()):
                for group in sorted(by_group):
                    if by_group[group] and len(class_samples) < max_videos_per_class:
                        class_samples.append(by_group[group].pop(0))
        samples.extend(class_samples)
    if not samples:
        raise FileNotFoundError(f"No videos found under {root}/alert and {root}/drowsy.")
    return samples


def stratified_group_split(
    samples: list[VideoSample],
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[VideoSample]]:
    grouped: dict[int, dict[str, list[VideoSample]]] = {0: {}, 1: {}}
    for sample in samples:
        grouped[sample.label].setdefault(sample.group, []).append(sample)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label, by_group in grouped.items():
        groups = list(by_group.keys())
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, int(round(n * test_size))) if n >= 3 and test_size > 0 else 0
        n_val = max(1, int(round(n * val_size))) if n >= 3 and val_size > 0 else 0
        if n_test + n_val >= n:
            n_val = max(0, n - n_test - 1)
        for group in groups[:n_test]:
            splits["test"].extend(by_group[group])
        for group in groups[n_test : n_test + n_val]:
            splits["val"].extend(by_group[group])
        for group in groups[n_test + n_val :]:
            splits["train"].extend(by_group[group])

    for split_samples in splits.values():
        rng.shuffle(split_samples)
    return splits


def read_video_clip(path: Path, num_frames: int, image_size: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_indices = list(range(num_frames))
    else:
        frame_indices = np.linspace(0, max(0, frame_count - 1), num_frames).astype(int).tolist()

    frames: list[np.ndarray] = []
    last: np.ndarray | None = None
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            if last is None:
                frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                frame = last.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        last = frame
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()

    clip = np.stack(frames, axis=0)
    clip = (clip - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(clip, (0, 3, 1, 2)).astype(np.float32)


class VideoDataset(Dataset):
    def __init__(self, samples: list[VideoSample], num_frames: int, image_size: int) -> None:
        self.samples = samples
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        clip = read_video_clip(sample.path, self.num_frames, self.image_size)
        return torch.from_numpy(clip), torch.tensor(sample.label, dtype=torch.long)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
        y_prob.extend(prob.tolist())

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    y_prob_np = np.asarray(y_prob)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).astype(int).tolist(),
    }
    metrics["auc"] = float(roc_auc_score(y_true_np, y_prob_np)) if len(np.unique(y_true_np)) == 2 else None
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Train a raw cropped-face Video Transformer.")
    p.add_argument("--root", type=str, default="data/raw/videos/uta_rldd_len10")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--image_size", type=int, default=112)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default="outputs/runs/video_transformer")
    p.add_argument("--max_videos_per_class", type=int, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(root, args.max_videos_per_class)
    splits = stratified_group_split(samples, args.val_size, args.test_size, args.seed)
    print(f"Device: {device}")
    for split, split_samples in splits.items():
        labels, counts = np.unique([s.label for s in split_samples], return_counts=True)
        label_summary = {LABEL_TO_CLASS[int(label)]: int(count) for label, count in zip(labels, counts)}
        print(f"{split:5s}: {len(split_samples)} videos, {len(set(s.group for s in split_samples))} subjects, {label_summary}")

    loaders = {
        split: DataLoader(
            VideoDataset(split_samples, args.num_frames, args.image_size),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        for split, split_samples in splits.items()
        if split_samples
    }

    model = FaceVideoTransformer(
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        layers=args.layers,
        dropout=args.dropout,
        num_frames=args.num_frames,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    ckpt_path = out_dir / "best.pt"
    history_path = out_dir / "history.csv"
    best_f1 = -1.0
    rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for x, y in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            seen += y.size(0)

        train_metrics = evaluate(model, loaders["train"], device)
        val_metrics = evaluate(model, loaders["val"], device) if "val" in loaders else train_metrics
        if float(val_metrics["f1"]) > best_f1:
            best_f1 = float(val_metrics["f1"])
            torch.save({"model_state": model.state_dict(), "args": vars(args), "class_to_label": CLASS_TO_LABEL}, ckpt_path)

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
        }
        rows.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={row['train_loss']:.4f} "
            f"train_f1={row['train_f1']:.3f} val_f1={row['val_f1']:.3f}"
        )

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics = {split: evaluate(model, loader, device) for split, loader in loaders.items()}
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    summary_path = out_dir / "summary_metrics.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "split", "accuracy", "precision", "recall", "f1", "auc"])
        writer.writeheader()
        for split, split_metrics in metrics.items():
            writer.writerow({"model": "face_video_transformer", "split": split, **{k: split_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "auc"]}})

    for split, split_metrics in metrics.items():
        cm = np.asarray(split_metrics["confusion_matrix"], dtype=int)
        np.savetxt(out_dir / f"{split}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Report table: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
