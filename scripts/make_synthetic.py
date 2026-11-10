"""Synthetic side-view stick-figure sessions for smoke-testing the whole pipeline before real
data exists. NOT for reporting results.

    python scripts/make_synthetic.py --participants 12
    python scripts/make_splits.py --config configs/synthetic.yaml
    python -m posetcn.train --config configs/synthetic.yaml --name smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from posetcn.config import ROOT, load_config, resolve  # noqa: E402
from posetcn.labels import frames_to_intervals, write_intervals  # noqa: E402

# tag mix drawn per segment (mirrors the recording protocol)
TAGS = ["good", "good", "forward_head", "rounded_shoulders", "slouched", "perched_edge",
        "forward_head+rounded_shoulders", "forward_head+rounded_shoulders+slouched", "forward_head+perched_edge",
        "transient"]
W_IMG, H_IMG, FPS = 1280, 720, 15


def unit(deg):
    r = np.radians(deg)
    return np.stack([np.sin(r), np.cos(r)], -1)  # angle from vertical, + = forward


def smooth_noise(rng, n, sd, alpha=0.97):
    z = np.zeros(n)
    e = rng.normal(0, sd * np.sqrt(1 - alpha ** 2), n)
    for i in range(1, n):
        z[i] = alpha * z[i - 1] + e[i]
    return z


def pose_params(tag, rng, person):
    """tag -> target [trunk, neck(rel), shoulder_shift, shrink, thigh, hip_slide] for one segment."""
    p = set(tag.split("+"))
    trunk = rng.normal(0, 3) + person["trunk_bias"]
    rel_neck = rng.normal(8, 4) + person["neck_bias"]
    sh, shrink, thigh, slide = rng.normal(0, 0.02), 1.0, rng.normal(0, 3), 0.0
    if "forward_head" in p:
        rel_neck += rng.normal(28, 6)
    if "rounded_shoulders" in p:
        sh += rng.normal(0.14, 0.03)
    if "slouched" in p:
        trunk += rng.normal(-12, 4)
        shrink = rng.normal(0.88, 0.03)
        slide = rng.normal(0.35, 0.08)
    if "perched_edge" in p:
        thigh += rng.normal(18, 4)
        trunk += rng.normal(6, 3)
    if "transient" in p:
        trunk += rng.normal(8, 8)
        rel_neck += rng.normal(5, 10)
    return np.array([trunk, rel_neck, sh, shrink, thigh, slide])


def session(rng, person, minutes):
    segs, T = [], 0
    while T < minutes * 60 * FPS:
        tag = str(rng.choice(TAGS))
        n = int(rng.uniform(5, 30) * FPS) if tag == "transient" else int(rng.uniform(20, 60) * FPS)
        segs.append((tag, n))
        T += n
    tags = np.concatenate([np.array([tag] * n, dtype=object) for tag, n in segs])
    tgt = np.concatenate([np.tile(pose_params(tag, rng, person), (n, 1)) for tag, n in segs])
    k = int(1.5 * FPS)  # smooth 1.5 s transitions
    kernel = np.ones(k) / k
    tgt = np.stack([np.convolve(np.pad(tgt[:, j], (k - 1, 0), mode="edge"), kernel, "valid")
                    for j in range(tgt.shape[1])], -1)
    t = np.arange(T) / FPS
    is_tr = np.array([tg == "transient" for tg in tags])
    trunk = tgt[:, 0] + smooth_noise(rng, T, 2.0) + is_tr * 10 * np.sin(2 * np.pi * 0.4 * t)
    neck = trunk + tgt[:, 1] + smooth_noise(rng, T, 3.0)
    sh, shrink, thigh, slide = tgt[:, 2], tgt[:, 3], tgt[:, 4] + smooth_noise(rng, T, 1.5), tgt[:, 5]
    arm = 25 + smooth_noise(rng, T, 5) + is_tr * 50 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)) + 60 * sh

    L0 = person["torso"]
    L = L0 * shrink
    Hp = np.stack([slide * L0, np.zeros(T)], -1)
    S = Hp + L[:, None] * unit(trunk) + np.stack([sh * L0, np.zeros(T)], -1)
    E = S + 0.45 * L0 * unit(neck)
    N = E + 0.18 * L0 * unit(np.full(T, 100.0))
    K = Hp + 0.9 * L0 * unit(90 + thigh)
    A = K + 0.9 * L0 * unit(np.full(T, 175.0) + thigh * 1.5)
    El = S + 0.6 * L0 * unit(180 - arm)
    Wr = El + 0.55 * L0 * unit(np.full(T, 80.0) + is_tr * 40 * np.sin(t))

    facing = rng.choice([-1, 1])
    side = rng.choice(["left", "right"])
    near = {"left": [0, 7, 11, 13, 15, 23, 25, 27], "right": [0, 8, 12, 14, 16, 24, 26, 28]}
    far = near["right" if side == "left" else "left"]
    pts = [N, E, S, El, Wr, Hp, K, A]
    origin = np.array([0.5 * W_IMG + rng.normal(0, 60), 0.6 * H_IMG + rng.normal(0, 30)])

    lm = np.zeros((T, 33, 4))
    lm[..., 3] = 0.1
    for idx_list, vis, off in ((far, 0.4, 0.03), (near[side], 0.95, 0.0)):
        for j, p in zip(idx_list, pts):
            lm[:, j, 0] = (origin[0] + facing * p[:, 0]) / W_IMG + off + rng.normal(0, 0.003, T)
            lm[:, j, 1] = (origin[1] - p[:, 1]) / H_IMG + rng.normal(0, 0.003, T)
            lm[:, j, 3] = np.clip(vis + rng.normal(0, 0.05, T), 0, 1)
    lm[rng.random(T) < 0.02] = np.nan

    # leave ~2 s around each change unlabeled, as real labels would
    lab = tags.copy()
    for c in np.where(tags[1:] != tags[:-1])[0] + 1:
        lab[max(0, c - FPS // 2):c + 2 * FPS] = ""
    return lm.astype(np.float32), t, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/synthetic.yaml"))
    ap.add_argument("--participants", type=int, default=12)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    lm_dir, lab_dir = resolve(cfg["paths"]["landmarks_dir"]), resolve(cfg["paths"]["labels_dir"])
    lm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.participants):
        pid = f"syn{i:02d}"
        person = {"torso": rng.uniform(140, 220), "trunk_bias": rng.normal(0, 3), "neck_bias": rng.normal(0, 4)}
        for s in range(args.sessions):
            lm, t, tags = session(rng, person, args.minutes)
            name = f"{pid}_s{s}"
            np.savez_compressed(lm_dir / f"{name}.npz", lm=lm, world=np.zeros((len(t), 33, 3), np.float32), t=t,
                                fps=FPS, width=W_IMG, height=H_IMG, pid=pid, session=f"s{s}", video="synthetic")
            write_intervals(lab_dir / f"{name}.csv", frames_to_intervals(tags, t))
    print(f"wrote {args.participants * args.sessions} synthetic sessions to {lm_dir}")


if __name__ == "__main__":
    main()
