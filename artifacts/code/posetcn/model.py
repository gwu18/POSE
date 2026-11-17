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


class GatedTemporalBlock(nn.Module):
    """WaveNet-style gated temporal block."""
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        self.filter = CausalConv1d(cin, cout, k, dilation)
        self.gate = CausalConv1d(cin, cout, k, dilation)
        self.out = nn.Sequential(nn.Conv1d(cout, cout, 1), nn.Dropout(dropout))
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        return F.relu(self.out(h) + self.down(x))


class DepthwiseTemporalBlock(nn.Module):
    """Depthwise-separable causal temporal block."""
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        self.depthwise = nn.Conv1d(cin, cin, k, dilation=dilation, groups=cin)
        self.left_pad = (k - 1) * dilation
        self.pointwise = nn.Conv1d(cin, cout, 1)
        self.norm = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.pad(x, (self.left_pad, 0))
        h = self.pointwise(self.depthwise(h))
        return F.relu(self.drop(F.relu(self.norm(h))) + self.down(x))


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


class VariantTCN(PostureTCN):
    """Selectable compact TCN blocks for architecture benchmarking."""
    def __init__(self, num_features, num_classes, channels=64, levels=4,
                 kernel_size=3, dropout=0.2, variant="standard"):
        _Normalized.__init__(self, num_features)
        layers, cin = [], num_features
        block_cls = {
            "standard": TemporalBlock,
            "gated": GatedTemporalBlock,
            "depthwise": DepthwiseTemporalBlock,
        }.get(variant)
        if block_cls is None:
            raise ValueError(f"unknown TCN variant {variant}")
        for i in range(levels):
            layers.append(block_cls(cin, channels, kernel_size, 2 ** i, dropout))
            cin = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, num_classes)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)


class MultiScaleTCN(_Normalized):
    """Parallel causal convolutions at each level, fused by a 1x1 convolution."""
    def __init__(self, num_features, num_classes, channels=16, levels=3, kernel_size=3, dropout=0.1, **_):
        super().__init__(num_features)
        self.blocks, cin = nn.ModuleList(), num_features
        for i in range(levels):
            dil = 2 ** i
            branches = nn.ModuleList([CausalConv1d(cin, channels, 3, dil),
                                      CausalConv1d(cin, channels, 5, dil)])
            self.blocks.append(nn.ModuleDict({
                "branches": branches,
                "fuse": nn.Sequential(nn.Conv1d(2 * channels, channels, 1),
                                      nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout)),
                "down": nn.Conv1d(cin, channels, 1) if cin != channels else nn.Identity(),
            }))
            cin = channels
        self.head = nn.Linear(channels, num_classes)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)

    def _features(self, x):
        h = self.normalize(x).transpose(1, 2)
        for block in self.blocks:
            h = F.relu(block["fuse"](torch.cat([b(h) for b in block["branches"]], 1)) + block["down"](h))
        return h.transpose(1, 2)

    def forward_all(self, x):
        return self.head(self._features(x))

    def forward(self, x):
        return self.head(self._features(x)[:, -1])


class AttentionTCN(PostureTCN):
    """Causal TCN followed by lightweight causal self-attention."""
    def __init__(self, num_features, num_classes, channels=16, levels=3, kernel_size=3, dropout=0.1, **_):
        super().__init__(num_features, num_classes, channels, levels, kernel_size, dropout)
        self.attn = nn.MultiheadAttention(channels, num_heads=1, dropout=dropout, batch_first=True)

    def _features(self, x):
        h = self.tcn(self.normalize(x).transpose(1, 2)).transpose(1, 2)
        n = h.shape[1]
        mask = torch.triu(torch.ones(n, n, device=h.device, dtype=torch.bool), diagonal=1)
        return self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]

    def forward_all(self, x):
        return self.head(self._features(x))

    def forward(self, x):
        return self.head(self._features(x)[:, -1])


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
        return VariantTCN(num_features, n_cls, m["channels"], m["levels"], m["kernel_size"], m["dropout"], m.get("variant", "standard"))
    if m["type"] == "multiscale_tcn":
        return MultiScaleTCN(num_features, n_cls, m["channels"], m["levels"], m["kernel_size"], m["dropout"])
    if m["type"] == "attention_tcn":
        return AttentionTCN(num_features, n_cls, m["channels"], m["levels"], m["kernel_size"], m["dropout"])
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
