from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from drowsiness_detection.data import FolderImageDataset
from drowsiness_detection.models.cnn import SmallCNN


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def main() -> int:
    p = argparse.ArgumentParser(description="Train a small CNN baseline on folder dataset.")
    p.add_argument("--data", type=str, default="data/processed", help="Root containing train/(alert|drowsy).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="outputs/runs/cnn_baseline.pt")
    args = p.parse_args()

    ds = FolderImageDataset(args.data, split="train", image_size=(224, 224))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)

    model = SmallCNN(num_classes=2).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        accs = []
        for x_np, y_np in dl:
            x = x_np.to(args.device, dtype=torch.float32)
            y = y_np.to(args.device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            accs.append(accuracy(logits.detach(), y))

        print(f"epoch {epoch:02d} | loss={sum(losses)/len(losses):.4f} | acc={sum(accs)/len(accs):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

