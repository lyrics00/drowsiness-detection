from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMLP(nn.Module):
    """
    Baseline temporal model for handcrafted features.

    Input: sequence of features (B,T,F)
    Output: logits (B,2)
    """

    def __init__(self, feat_dim: int, hidden: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean-pool over time then classify
        x = x.mean(dim=1)
        return self.net(x)


class LSTMClassifier(nn.Module):
    """
    Temporal encoder for sequences (e.g., EAR/MAR/pose or CNN embeddings).
    """

    def __init__(self, feat_dim: int, hidden: int = 128, num_layers: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

