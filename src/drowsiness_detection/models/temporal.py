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

    def __init__(self, feat_dim: int, hidden: int = 64, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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

    def __init__(
        self,
        feat_dim: int,
        hidden: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.norm = nn.LayerNorm(hidden * (2 if bidirectional else 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(self.norm(last)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer encoder over a sequence of per-frame video features.

    This is the temporal-modeling upgrade requested in the proposal feedback:
    each input row is a video frame descriptor (EAR, MAR, head pose, optional
    deltas/CNN probabilities), and self-attention learns which frames in the
    clip matter for the drowsiness decision.
    """

    def __init__(
        self,
        feat_dim: int,
        embed_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_len: int = 600,
        pool: str = "cls",
    ):
        super().__init__()
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be 'cls' or 'mean'")

        self.pool = pool
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)  # (B, T, D)
        if self.pool == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)  # (B, T or T+1, D)
        if self.pool == "cls":
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        x = self.dropout(self.norm(x))
        logits = self.head(x)  # (B, num_classes)
        return logits

