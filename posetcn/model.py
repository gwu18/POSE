"""Causal dilated TCN (Bai et al., 2018) + single-frame MLP baseline.

Both models take raw features (B, T, F) and normalise internally with stored
mean/std buffers, so the exported model needs no extra preprocessing step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, cin, cout, k, dilation):
        super().__init__(cin, cout, k, dilation=dilation)
        self.left_pad = (k - 1) * dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(cin, cout, k, dilation), nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(cout, cout, k, dilation), nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.down(x))


class _Normalized(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(num_features))
        self.register_buffer("feat_std", torch.ones(num_features))

    def set_norm(self, mean, std):
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-3))

    def normalize(self, x):
        return (x - self.feat_mean) / self.feat_std


class PostureTCN(_Normalized):
    def __init__(self, num_features, num_classes, channels=64, levels=4, kernel_size=3, dropout=0.2):
        super().__init__(num_features)
        layers, cin = [], num_features
        for i in range(levels):
            layers.append(TemporalBlock(cin, channels, kernel_size, 2 ** i, dropout))
            cin = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, num_classes)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)

    def forward_all(self, x):
        """(B,T,F) -> per-timestep logits (B,T,C). Causal: step t sees only <= t."""
        h = self.tcn(self.normalize(x).transpose(1, 2))
        return self.head(h.transpose(1, 2))

    def forward(self, x):
        """(B,T,F) -> logits for the last timestep (B,C)."""
        h = self.tcn(self.normalize(x).transpose(1, 2))
        return self.head(h[:, :, -1])


class FrameMLP(_Normalized):
    """Baseline: classifies from the LAST frame only (no temporal context)."""

    def __init__(self, num_features, num_classes, channels=64, dropout=0.2, **_):
        super().__init__(num_features)
        self.net = nn.Sequential(
            nn.Linear(num_features, channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(channels, channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )
        self.receptive_field = 1

    def forward_all(self, x):
        return self.net(self.normalize(x))

    def forward(self, x):
        return self.net(self.normalize(x[:, -1]))


def build_model(cfg: dict, num_features: int) -> nn.Module:
    m = cfg["model"]
    n_cls = len(cfg["classes"])
    if m["type"] == "tcn":
        return PostureTCN(num_features, n_cls, m["channels"], m["levels"], m["kernel_size"], m["dropout"])
    if m["type"] == "frame_mlp":
        return FrameMLP(num_features, n_cls, m["channels"], m["dropout"])
    raise ValueError(f"unknown model type {m['type']}")


def load_run(run_dir, device="cpu"):
    """Load best checkpoint from a run dir -> (model, cfg, ckpt)."""
    from pathlib import Path
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, ckpt["num_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg, ckpt
