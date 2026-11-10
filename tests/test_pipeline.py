"""Core invariants. Run: pytest -q   (or: python tests/test_pipeline.py)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from posetcn.config import load_config
from posetcn.alerts import AlertEngine
from posetcn.dataset import DELTA_SLICE, WindowDataset, make_clip, window_target
from posetcn.features import NUM_FEATURES, canonical_points, geometry, points_to_features
from posetcn.labels import frame_targets, frames_to_intervals, normalize_tag
from posetcn.model import FrameMLP, PostureTCN

CFG = load_config()


def fake_landmarks(T=60, trunk=0.0, neck=10.0, facing=1, seed=0, thigh=0.0):
    """Side-view stick figure in normalised image coords, left side visible."""
    rng = np.random.default_rng(seed)
    lm = np.zeros((T, 33, 4))
    lm[..., 3] = 0.2
    L = 0.25
    u = lambda d: np.array([np.sin(np.radians(d)), np.cos(np.radians(d))])  # noqa: E731
    H = np.array([0.0, 0.0])
    S = H + L * u(trunk)
    E = S + 0.45 * L * u(neck)
    pts = {0: E + 0.05 * u(100), 7: E, 11: S, 13: S + 0.15 * u(160), 15: S + 0.2 * u(100), 23: H,
           25: H + 0.22 * u(90 + thigh), 27: H + 0.22 * u(90 + thigh) + 0.22 * u(180)}
    for j, p in pts.items():
        lm[:, j, 0] = 0.5 + facing * p[0] + rng.normal(0, 1e-4, T)
        lm[:, j, 1] = 0.7 - p[1] + rng.normal(0, 1e-4, T)
        lm[:, j, 3] = 0.95
    return lm


def test_angles_recovered():
    lm = fake_landmarks(trunk=20, neck=40, thigh=15)
    pts, vis, det, meta = canonical_points(lm, 1000, 1000)
    g = geometry(pts)
    assert meta["side"] == "left"
    assert abs(np.median(g[:, 0]) - 40) < 1 and abs(np.median(g[:, 1]) - 20) < 1
    assert abs(np.median(g[:, 8]) - 15) < 1          # thigh slope (perched edge)
    assert abs(np.median(g[:, 9]) - 105) < 1         # knee angle


def test_facing_and_scale_invariance():
    a = fake_landmarks(facing=1, trunk=15, neck=35)
    b = fake_landmarks(facing=-1, trunk=15, neck=35)
    fa = points_to_features(*canonical_points(a, 1000, 1000)[:3])
    pb, vb, db, _ = canonical_points(b, 1000, 1000)
    fb = points_to_features(pb, vb, db)
    fc = points_to_features(pb * 2.5 + 37.0, vb, db)  # scale + translate
    assert fa.shape == (60, NUM_FEATURES)
    assert np.allclose(fa[:, :11], fb[:, :11], atol=1e-2)  # mirrored person -> same geometry
    assert np.allclose(fb, fc, atol=1e-3)


def test_missing_frames_filled():
    lm = fake_landmarks()
    lm[10:15] = np.nan
    pts, vis, det, _ = canonical_points(lm, 1000, 1000)
    assert np.isfinite(pts).all() and det[10:15].sum() == 0 and vis[12].sum() == 0
    assert np.isfinite(points_to_features(pts, vis, det)).all()


def test_multilabel_targets():
    C = CFG["classes"]
    t = np.arange(200) / 10
    iv = [(1, 4, "good"), (5, 9, normalize_tag("slouched+forward_head")), (10, 12, "transient"), (13, 15, "free")]
    Y, M, tags = frame_targets(iv, t, CFG)
    fh, rs, sl, tr = (C.index(c) for c in ("forward_head", "rounded_shoulders", "slouched", "transient"))
    assert M[0].sum() == 0                                  # unlabeled
    assert Y[20].sum() == 0 and M[20].all()                 # good: all known zero
    assert Y[60, fh] == 1 and Y[60, sl] == 1 and Y[60, rs] == 0 and M[60].all()
    assert Y[110, tr] == 1 and M[110, tr] == 1 and M[110, fh] == 0   # faults unknown while transient
    assert M[140].sum() == 0                                # 'free' is not trainable
    assert [b[2] for b in frames_to_intervals(tags, t)] == ["good", "forward_head+slouched", "transient", "free"]


def test_window_target_purity():
    Y = np.zeros((15, 5), np.float32)
    M = np.ones((15, 5), np.float32)
    Y[:10, 0] = 1   # 10/15 positive -> not pure enough at 0.8
    Y[:, 1] = 1
    y, m = window_target(Y, M, 0.8)
    assert m[0] == 0 and y[1] == 1 and m[1] == 1 and y[2] == 0 and m[2] == 1


def _clip():
    lm = np.concatenate([fake_landmarks(150, 0, 10), fake_landmarks(150, 20, 35)])
    t = np.arange(300) / 15
    return make_clip("p00_s0", "p00", t, lm, 1000, 1000, [(0, 10, "good"), (10, 20, "forward_head")], CFG)


def test_window_matches_streaming_features():
    c = _clip()
    ds = WindowDataset([c], CFG, augment=False)
    x, y, m = ds[5]
    _, e = ds.index[5]
    W = CFG["data"]["window"]
    direct = points_to_features(c.pts[e - W + 1:e + 1], c.vis[e - W + 1:e + 1], c.det[e - W + 1:e + 1], c.ref)
    assert np.allclose(x.numpy(), direct, atol=1e-5)
    assert (x[0, DELTA_SLICE] == 0).all()
    fh = CFG["classes"].index("forward_head")
    assert ds.positives()[fh] > 0 and ds.knowns()[fh] > ds.positives()[fh]


def test_augmented_window_shape():
    ds = WindowDataset([_clip()], CFG, augment=True)
    x, _, _ = ds[3]
    assert x.shape == (CFG["data"]["window"], NUM_FEATURES) and torch.isfinite(x).all()


def test_tcn_is_causal_and_shapes():
    torch.manual_seed(0)
    m = PostureTCN(NUM_FEATURES, 5).eval()
    x = torch.randn(2, 45, NUM_FEATURES)
    assert m(x).shape == (2, 5) and m.forward_all(x).shape == (2, 45, 5)
    y1 = m.forward_all(x)
    x2 = x.clone()
    x2[:, 30:] = torch.randn_like(x2[:, 30:])
    y2 = m.forward_all(x2)
    assert torch.allclose(y1[:, :30], y2[:, :30], atol=1e-5)  # future can't change the past
    assert m.receptive_field == 61
    assert FrameMLP(NUM_FEATURES, 5)(x).shape == (2, 5)


def test_alerts_independent_and_gated():
    cfg = {**CFG, "alerts": {**CFG["alerts"], "persist_s": 1.0, "cooldown_s": 100.0}}
    eng = AlertEngine(cfg)
    C = cfg["classes"]
    p = np.zeros(len(C))
    p[C.index("forward_head")] = p[C.index("rounded_shoulders")] = 0.9
    fired = [a for k in range(20) for a in eng.update(k * 0.1, p)]
    assert sorted(a.zone for a in fired) == ["neck", "shoulders"]       # both buzz, once each
    eng.reset()
    p[C.index("transient")] = 0.9
    assert not [a for k in range(20) for a in eng.update(k * 0.1, p)]  # gated while transient


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
