"""Clips (cached landmarks + labels) -> sliding-window multi-label training samples."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import resolve
from .features import GEOM_NAMES, HIP, NUM_FEATURES, canonical_points, points_to_features, reference_scale
from .labels import frame_targets, read_intervals

DELTA_SLICE = slice(NUM_FEATURES - len(GEOM_NAMES), NUM_FEATURES)


@dataclass
class Clip:
    name: str
    pid: str
    t: np.ndarray            # (T,) seconds
    pts: np.ndarray          # (T,8,2) canonical points
    vis: np.ndarray          # (T,8)
    det: np.ndarray          # (T,)
    ref: float               # reference torso length
    Y: np.ndarray            # (T,C) targets in {0,1}
    M: np.ndarray            # (T,C) mask, 1 = known
    tags: np.ndarray         # (T,) tag strings, '' = unlabeled
    feats: np.ndarray = field(default=None)  # (T,F) un-augmented features
    meta: dict = field(default_factory=dict)


def make_clip(name, pid, t, lm, width, height, intervals, cfg) -> Clip:
    pts, vis, det, meta = canonical_points(lm, width, height)
    ref = reference_scale(pts, det)
    Y, M, tags = frame_targets(intervals, t, cfg)
    clip = Clip(name, pid, t, pts, vis, det, ref, Y, M, tags, meta=meta)
    clip.feats = points_to_features(pts, vis, det, ref)
    return clip


def load_clip(npz_path: Path, labels_dir: Path, cfg: dict) -> Clip | None:
    d = np.load(npz_path, allow_pickle=False)
    name = npz_path.stem
    label_path = labels_dir / f"{name}.csv"
    if not label_path.exists():
        warnings.warn(f"no labels for {name}; skipping")
        return None
    pid = str(d["pid"]) if "pid" in d else name.split("_")[0]
    return make_clip(name, pid, d["t"], d["lm"], int(d["width"]), int(d["height"]),
                     read_intervals(label_path), cfg)


def load_clips(cfg: dict, pids: list[str] | None = None) -> list[Clip]:
    lm_dir, lab_dir = resolve(cfg["paths"]["landmarks_dir"]), resolve(cfg["paths"]["labels_dir"])
    clips = []
    for p in sorted(lm_dir.glob("*.npz")):
        if pids is not None and p.stem.split("_")[0] not in pids:
            continue
        c = load_clip(p, lab_dir, cfg)
        if c is not None and (pids is None or c.pid in pids):
            clips.append(c)
    return clips


def load_splits(cfg: dict) -> dict[str, list[str]]:
    with open(resolve(cfg["paths"]["splits"])) as f:
        return json.load(f)


def window_target(Y: np.ndarray, M: np.ndarray, purity: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-output majority over the tail frames. An output is known only if >= purity of the tail
    frames are labeled for it and agree."""
    n = len(Y)
    known = M.sum(0)
    pos = (Y * M).sum(0)
    y = (pos >= known / 2).astype(np.float32) * (known > 0)
    agree = np.where(y > 0, pos, known - pos)
    m = (agree >= purity * n).astype(np.float32)
    return y, m


class WindowDataset(Dataset):
    """Items: x (W,F) float, y (C,) {0,1}, m (C,) mask."""

    def __init__(self, clips: list[Clip], cfg: dict, augment: bool = False, stride: int | None = None):
        self.clips, self.cfg = clips, cfg
        self.augment = augment and cfg["augment"]["enabled"]
        dc = cfg["data"]
        self.W, self.tail = dc["window"], dc["label_tail"]
        stride = stride or dc["stride"]
        self.index: list[tuple[int, int]] = []
        ys, ms = [], []
        for ci, c in enumerate(clips):
            for e in range(self.W - 1, len(c.t), stride):
                y, m = window_target(c.Y[e - self.tail + 1:e + 1], c.M[e - self.tail + 1:e + 1], dc["tail_purity"])
                if m.sum() == 0 or c.det[e - self.W + 1:e + 1].mean() < dc["min_detected"]:
                    continue
                self.index.append((ci, e))
                ys.append(y)
                ms.append(m)
        C = len(cfg["classes"])
        self.Y = np.stack(ys) if ys else np.zeros((0, C), np.float32)
        self.M = np.stack(ms) if ms else np.zeros((0, C), np.float32)
        self.rng = np.random.default_rng(cfg["train"]["seed"])

    def positives(self) -> np.ndarray:
        return (self.Y * self.M).sum(0)

    def knowns(self) -> np.ndarray:
        return self.M.sum(0)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ci, e = self.index[i]
        c = self.clips[ci]
        if self.augment:
            x = self._augmented(c, e)
        else:
            x = c.feats[e - self.W + 1:e + 1].copy()
            x[0, DELTA_SLICE] = 0.0  # identical to computing features on this window alone
        return torch.from_numpy(x), torch.from_numpy(self.Y[i]), torch.from_numpy(self.M[i])

    # ---------------- augmentation (operates on canonical points) ----------------
    def _augmented(self, c: Clip, e: int) -> np.ndarray:
        a, rng, W = self.cfg["augment"], self.rng, self.W
        s = rng.uniform(*a["time_stretch"])
        L = int(np.clip(round(W * s), 2, e + 1))
        src = np.linspace(e - L + 1, e, W)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, e)
        w = (src - lo)[:, None, None]
        pts = c.pts[lo] * (1 - w) + c.pts[hi] * w
        near = np.round(src).astype(int)
        vis, det = c.vis[near].copy(), c.det[near].copy()

        # camera roll about the hip + foreshortening in x
        th = np.radians(rng.uniform(-a["rotate_deg"], a["rotate_deg"]))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
        hip = pts[:, HIP:HIP + 1]
        pts = (pts - hip) @ R.T
        pts[..., 0] *= rng.uniform(*a["x_squash"])
        pts = pts + hip
        pts = pts + rng.normal(0, a["noise_std"] * c.ref, pts.shape)

        # simulated dropouts: hold the previous pose, zero visibility
        drop = rng.random(W) < a["vis_drop_prob"]
        drop[0] = False
        for k in np.where(drop)[0]:
            pts[k] = pts[k - 1]
            vis[k] = 0.0
            det[k] = 0.0
        return points_to_features(pts.astype(np.float32), vis, det, c.ref)


def feature_stats(ds: WindowDataset, max_samples: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of features over (a sample of) un-augmented training windows."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(len(ds), max_samples), replace=False)
    aug, ds.augment = ds.augment, False
    X = np.concatenate([ds[i][0].numpy() for i in idx])
    ds.augment = aug
    return X.mean(0), X.std(0) + 1e-6
