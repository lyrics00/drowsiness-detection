from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from drowsiness_detection.models.temporal import LSTMClassifier


def make_windows(x: np.ndarray, y: int, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(0, max(0, len(x) - window + 1), stride):
        xs.append(x[i : i + window])
        ys.append(y)
    if not xs:
        return np.empty((0, window, x.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def main() -> int:
    """
    Train a simple LSTM on feature CSVs produced by `extract_features_video.py`.

    Expected folder layout:
      data/processed/features/alert/*.csv
      data/processed/features/drowsy/*.csv
    """
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="data/processed/features")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="outputs/runs/lstm_features.pt")
    args = p.parse_args()

    root = Path(args.root)
    files = []
    for cls, label in [("alert", 0), ("drowsy", 1)]:
        for pth in sorted((root / cls).glob("*.csv")):
            files.append((pth, label))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {root}/(alert|drowsy).")

    Xs = []
    Ys = []
    for pth, label in files:
        df = pd.read_csv(pth)
        feats = df[["ear", "mar", "yaw_deg", "pitch_deg", "roll_deg"]].to_numpy(dtype=np.float32)
        xw, yw = make_windows(feats, label, window=args.window, stride=args.stride)
        if len(xw) > 0:
            Xs.append(xw)
            Ys.append(yw)

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    print(f"Windows: {len(X)}  (T={args.window}, F={X.shape[-1]})")

    x = torch.from_numpy(X).to(args.device)
    y = torch.from_numpy(Y).to(args.device)

    model = LSTMClassifier(feat_dim=X.shape[-1], hidden=128, num_layers=1, num_classes=2).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        acc = float((logits.argmax(dim=1) == y).float().mean().item())
        print(f"epoch {epoch:02d} | loss={loss.item():.4f} | acc={acc:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

