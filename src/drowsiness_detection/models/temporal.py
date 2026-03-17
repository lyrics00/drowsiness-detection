from __future__ import annotations

import math
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, feat_dim: int, embed_dim: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)            # (B, T, D)
        x = self.pos_enc(x)
        x = self.encoder(x)               # (B, T, D)
        x = x.mean(dim=1)                 # mean-pool over time -> (B, D)
        logits = self.head(x)             # (B, num_classes)
        return logits

