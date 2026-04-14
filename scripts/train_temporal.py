from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from drowsiness_detection.models.temporal import LSTMClassifier, TemporalMLP, TransformerClassifier


DEFAULT_FEATURES = ["ear", "mar", "yaw_deg", "pitch_deg", "roll_deg"]
CLASS_TO_LABEL = {"alert": 0, "drowsy": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    label: int
    group: str
    split: str | None = None


@dataclass
class WindowBundle:
    x: np.ndarray
    y: np.ndarray
    groups: list[str]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


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


def parse_feature_columns(raw: str) -> list[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("At least one feature column is required.")
    return cols


def infer_group(path: Path, class_dir: Path) -> str:
    rel = path.relative_to(class_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return path.stem


def discover_records(root: Path) -> dict[str, list[VideoRecord]]:
    """
    Accept either:
      data/processed/features/{alert,drowsy}/*.csv
    or explicit splits:
      data/processed/features/{train,val,test}/{alert,drowsy}/*.csv
    """
    split_dirs = ["train", "val", "test"]
    has_explicit_splits = any((root / split).exists() for split in split_dirs)

    records: dict[str, list[VideoRecord]] = {"all": [], "train": [], "val": [], "test": []}
    if has_explicit_splits:
        for split in split_dirs:
            for cls_name, label in CLASS_TO_LABEL.items():
                cls_dir = root / split / cls_name
                for path in sorted(cls_dir.rglob("*.csv")):
                    records[split].append(
                        VideoRecord(path=path, label=label, group=infer_group(path, cls_dir), split=split)
                    )
        return records

    for cls_name, label in CLASS_TO_LABEL.items():
        cls_dir = root / cls_name
        for path in sorted(cls_dir.rglob("*.csv")):
            records["all"].append(VideoRecord(path=path, label=label, group=infer_group(path, cls_dir)))
    return records


def stratified_group_split(
    records: list[VideoRecord],
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[VideoRecord]]:
    by_label_group: dict[int, dict[str, list[VideoRecord]]] = {0: {}, 1: {}}
    for rec in records:
        by_label_group[rec.label].setdefault(rec.group, []).append(rec)

    rng = random.Random(seed)
    split_records = {"train": [], "val": [], "test": []}
    for label, grouped_records in by_label_group.items():
        groups = list(grouped_records.keys())
        rng.shuffle(groups)
        n = len(groups)
        if n == 0:
            continue

        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        if n >= 3 and test_size > 0:
            n_test = max(1, n_test)
        if n >= 3 and val_size > 0:
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            overflow = n_test + n_val - (n - 1)
            n_val = max(0, n_val - overflow)

        for group in groups[:n_test]:
            split_records["test"].extend(grouped_records[group])
        for group in groups[n_test : n_test + n_val]:
            split_records["val"].extend(grouped_records[group])
        for group in groups[n_test + n_val :]:
            split_records["train"].extend(grouped_records[group])

    for split in split_records.values():
        rng.shuffle(split)
    return split_records


def load_feature_matrix(path: Path, feature_cols: list[str], include_deltas: bool) -> np.ndarray:
    df = pd.read_csv(path)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing feature columns: {missing}")
    if "frame" in df.columns:
        df = df.sort_values("frame")

    feats = df[feature_cols].astype(np.float32)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    x = feats.to_numpy(dtype=np.float32)

    if include_deltas:
        deltas = np.diff(x, axis=0, prepend=x[:1])
        x = np.concatenate([x, deltas.astype(np.float32)], axis=1)
    return x


def make_windows(
    x: np.ndarray,
    label: int,
    group: str,
    window: int,
    stride: int,
    pad_short: bool,
) -> WindowBundle:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[str] = []

    if len(x) < window:
        if not pad_short or len(x) == 0:
            return WindowBundle(
                x=np.empty((0, window, x.shape[1]), dtype=np.float32),
                y=np.empty((0,), dtype=np.int64),
                groups=[],
            )
        pad = np.repeat(x[-1:], window - len(x), axis=0)
        xs.append(np.concatenate([x, pad], axis=0))
        ys.append(label)
        groups.append(group)
    else:
        for start in range(0, len(x) - window + 1, stride):
            xs.append(x[start : start + window])
            ys.append(label)
            groups.append(group)

    return WindowBundle(
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        groups=groups,
    )


def build_split_windows(
    records: list[VideoRecord],
    feature_cols: list[str],
    include_deltas: bool,
    window: int,
    stride: int,
    pad_short: bool,
) -> WindowBundle:
    bundles: list[WindowBundle] = []
    feat_dim = len(feature_cols) * (2 if include_deltas else 1)

    for rec in records:
        x = load_feature_matrix(rec.path, feature_cols, include_deltas)
        bundles.append(make_windows(x, rec.label, rec.group, window, stride, pad_short))

    non_empty = [b for b in bundles if len(b.y) > 0]
    if not non_empty:
        return WindowBundle(
            x=np.empty((0, window, feat_dim), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            groups=[],
        )
    return WindowBundle(
        x=np.concatenate([b.x for b in non_empty], axis=0),
        y=np.concatenate([b.y for b in non_empty], axis=0),
        groups=[g for b in non_empty for g in b.groups],
    )


def normalize_splits(splits: dict[str, WindowBundle]) -> tuple[dict[str, WindowBundle], np.ndarray, np.ndarray]:
    train_x = splits["train"].x
    if len(train_x) == 0:
        raise ValueError("No training windows were created. Check CSV length, --window, and --stride.")

    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = flat.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    normalized: dict[str, WindowBundle] = {}
    for split, bundle in splits.items():
        if len(bundle.x) == 0:
            normalized[split] = bundle
            continue
        x = ((bundle.x - mean) / std).astype(np.float32)
        normalized[split] = WindowBundle(x=x, y=bundle.y, groups=bundle.groups)
    return normalized, mean.squeeze(0), std.squeeze(0)


def build_model(model_name: str, feat_dim: int, args: argparse.Namespace) -> nn.Module:
    if model_name == "transformer":
        return TransformerClassifier(
            feat_dim=feat_dim,
            embed_dim=args.embed_dim,
            nhead=args.nhead,
            num_layers=args.layers,
            num_classes=2,
            dropout=args.dropout,
            max_len=args.window,
            pool=args.pool,
        )
    if model_name == "lstm":
        return LSTMClassifier(
            feat_dim=feat_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            num_classes=2,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
    if model_name == "mlp":
        return TemporalMLP(feat_dim=feat_dim, hidden=args.hidden, num_classes=2, dropout=args.dropout)
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    all_y: list[int] = []
    all_pred: list[int] = []
    all_prob: list[float] = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_prob.extend(probs.tolist())
        all_pred.extend(preds.tolist())
        all_y.extend(y.numpy().tolist())

    if not all_y:
        return {}

    y_true = np.asarray(all_y)
    y_pred = np.asarray(all_pred)
    y_prob = np.asarray(all_prob)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = None
    return metrics


def train_one_model(
    model_name: str,
    splits: dict[str, WindowBundle],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    feature_cols: list[str],
) -> dict[str, object]:
    feat_dim = splits["train"].x.shape[-1]
    model = build_model(model_name, feat_dim, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        SequenceDataset(splits["train"].x, splits["train"].y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(SequenceDataset(splits["val"].x, splits["val"].y), batch_size=args.batch_size, shuffle=False)
        if len(splits["val"].y) > 0
        else None
    )
    test_loader = (
        DataLoader(SequenceDataset(splits["test"].x, splits["test"].y), batch_size=args.batch_size, shuffle=False)
        if len(splits["test"].y) > 0
        else None
    )

    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "best.pt"
    history_path = model_dir / "history.csv"

    best_score = -1.0
    best_epoch = 0
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * y.size(0)
            seen += y.size(0)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else train_metrics
        score = float(val_metrics.get("f1", 0.0))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "feature_cols": feature_cols,
                    "include_deltas": args.include_deltas,
                    "mean": mean,
                    "std": std,
                    "class_to_label": CLASS_TO_LABEL,
                    "args": vars(args),
                    "feat_dim": feat_dim,
                },
                ckpt_path,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, seen),
            "train_accuracy": train_metrics.get("accuracy"),
            "train_f1": train_metrics.get("f1"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_f1": val_metrics.get("f1"),
            "val_recall": val_metrics.get("recall"),
        }
        history_rows.append(row)
        print(
            f"{model_name:11s} epoch {epoch:03d}/{args.epochs} "
            f"loss={row['train_loss']:.4f} "
            f"train_f1={row['train_f1']:.3f} "
            f"val_f1={row['val_f1']:.3f}"
        )

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = {
        "model": model_name,
        "best_epoch": best_epoch,
        "train": evaluate(model, train_loader, device),
        "val": evaluate(model, val_loader, device) if val_loader is not None else None,
        "test": evaluate(model, test_loader, device) if test_loader is not None else None,
        "checkpoint": str(ckpt_path),
        "history": str(history_path),
    }

    with (model_dir / "metrics.json").open("w") as f:
        json.dump(final_metrics, f, indent=2)

    for split in ["train", "val", "test"]:
        split_metrics = final_metrics.get(split)
        if not split_metrics:
            continue
        cm = np.asarray(split_metrics["confusion_matrix"], dtype=int)
        np.savetxt(model_dir / f"{split}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    return final_metrics


def summarize_windows(splits: dict[str, WindowBundle], records: dict[str, list[VideoRecord]]) -> None:
    for split in ["train", "val", "test"]:
        labels, counts = np.unique(splits[split].y, return_counts=True) if len(splits[split].y) else ([], [])
        label_summary = {LABEL_TO_CLASS[int(label)]: int(count) for label, count in zip(labels, counts)}
        print(
            f"{split:5s}: {len(records[split])} videos, {len(set(r.group for r in records[split]))} groups, "
            f"{len(splits[split].y)} windows, labels={label_summary}"
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train report-grade temporal baselines and a Video Feature Transformer on per-frame CSVs."
    )
    p.add_argument("--root", type=str, default="data/processed/features")
    p.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    p.add_argument("--include_deltas", action="store_true", help="Append frame-to-frame differences for each feature.")
    p.add_argument("--window", type=int, default=90, help="Frames per temporal clip. 90 frames is about 3s at 30 FPS.")
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--pad_short", action="store_true")
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--models", type=str, default="transformer,lstm,mlp")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pool", choices=["cls", "mean"], default="cls")
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default="outputs/runs/temporal")
    args = p.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    root = Path(args.root)
    feature_cols = parse_feature_columns(args.features)
    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    records = discover_records(root)
    if records["all"]:
        split_records = stratified_group_split(records["all"], args.val_size, args.test_size, args.seed)
    else:
        split_records = {split: records[split] for split in ["train", "val", "test"]}

    if not any(split_records.values()):
        raise FileNotFoundError(
            f"No CSVs found under {root}. Expected {root}/alert/*.csv and {root}/drowsy/*.csv "
            f"or explicit {root}/train|val|test/alert|drowsy/*.csv folders."
        )

    splits = {
        split: build_split_windows(
            split_records[split],
            feature_cols=feature_cols,
            include_deltas=args.include_deltas,
            window=args.window,
            stride=args.stride,
            pad_short=args.pad_short,
        )
        for split in ["train", "val", "test"]
    }
    splits, mean, std = normalize_splits(splits)

    print(f"Device: {device}")
    print(f"Feature columns: {feature_cols}" + (" + deltas" if args.include_deltas else ""))
    print(f"Window={args.window} stride={args.stride}")
    summarize_windows(splits, split_records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "root": str(root),
        "feature_cols": feature_cols,
        "include_deltas": args.include_deltas,
        "window": args.window,
        "stride": args.stride,
        "class_to_label": CLASS_TO_LABEL,
        "splits": {
            split: [{"path": str(rec.path), "label": rec.label, "group": rec.group} for rec in split_records[split]]
            for split in ["train", "val", "test"]
        },
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    all_metrics = []
    for model_name in model_names:
        metrics = train_one_model(model_name, splits, args, device, out_dir, mean, std, feature_cols)
        all_metrics.append(metrics)

    summary = []
    for metrics in all_metrics:
        for split in ["train", "val", "test"]:
            split_metrics = metrics.get(split)
            if not split_metrics:
                continue
            summary.append(
                {
                    "model": metrics["model"],
                    "split": split,
                    "accuracy": split_metrics["accuracy"],
                    "precision": split_metrics["precision"],
                    "recall": split_metrics["recall"],
                    "f1": split_metrics["f1"],
                    "auc": split_metrics["auc"],
                }
            )

    summary_path = out_dir / "summary_metrics.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "split", "accuracy", "precision", "recall", "f1", "auc"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSaved temporal experiment artifacts to: {out_dir}")
    print(f"Report table: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
