"""Evaluate a trained run on held-out participants.

    python -m posetcn.evaluate runs/<name>              # test split
    python -m posetcn.evaluate runs/<name> --split val

Reports window-level multi-label metrics (per-output P/R/F1/AUROC, derived good-posture F1),
per-participant mean fault F1, and streaming alert metrics (false alerts per 10 min, latency).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .alerts import simulate
from .dataset import DELTA_SLICE, Clip, WindowDataset, load_clips
from .metrics import format_report, report
from .model import load_run
from .train import pick_device, predict


@torch.no_grad()
def clip_frame_probs(model, clip: Clip, window: int, n_classes: int, device, batch: int = 1024) -> np.ndarray:
    """Probabilities for every frame, as the live app would compute them (one window per frame).
    Frames before the first full window get prob 0."""
    T, C = len(clip.feats), n_classes
    probs = np.zeros((T, C), dtype=np.float32)
    ends = np.arange(window - 1, T)
    for s in range(0, len(ends), batch):
        e = ends[s:s + batch]
        x = np.stack([clip.feats[k - window + 1:k + 1] for k in e]).copy()
        x[:, 0, DELTA_SLICE] = 0.0
        probs[e] = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)

    device = pick_device()
    model, cfg, ckpt = load_run(args.run_dir, device)
    classes = cfg["classes"]
    pids = ckpt["splits"].get(args.split) or []
    clips = load_clips(cfg, pids)
    if not clips:
        raise SystemExit(f"no clips for split '{args.split}' (pids={pids})")

    ds = WindowDataset(clips, cfg, augment=False, stride=args.stride)
    Y, P, M = predict(model, DataLoader(ds, batch_size=1024), device)
    overall = report(Y, P, M, cfg)
    print(f"== {args.split}: {len(clips)} clips, {len(pids)} participants, {len(ds)} windows ==\n")
    print(format_report(overall, classes))

    # per participant
    pid_of = np.array([clips[ci].pid for ci, _ in ds.index])
    per_pid = {}
    print("\nper-participant mean fault F1:")
    for pid in sorted(set(pid_of)):
        m = pid_of == pid
        per_pid[pid] = report(Y[m], P[m], M[m], cfg)["mean_fault_f1"]
        print(f"  {pid:<10}{per_pid[pid]:.3f}")

    # streaming alert simulation
    agg = defaultdict(float)
    lat = []
    for c in clips:
        probs = clip_frame_probs(model, c, cfg["data"]["window"], len(classes), device)
        s = simulate(cfg, c.t, probs, c.Y, c.M)
        for k in ("alerts", "false_alerts", "labeled_minutes", "missed_segments"):
            agg[k] += s[k]
        lat += s["latencies"]
    fa10 = agg["false_alerts"] / max(agg["labeled_minutes"], 1e-6) * 10
    alerts = {**agg, "false_alerts_per_10min": fa10,
              "median_latency_s": float(np.median(lat)) if lat else None, "detected_segments": len(lat)}
    print(f"\nalerts: {int(agg['alerts'])} total, {int(agg['false_alerts'])} false "
          f"({fa10:.2f} per 10 min), fault runs caught {len(lat)}, missed {int(agg['missed_segments'])}, "
          f"median latency {alerts['median_latency_s']}")

    out = Path(args.run_dir) / f"eval_{args.split}.json"
    with open(out, "w") as f:
        json.dump({"overall": overall, "per_participant_mean_fault_f1": per_pid, "alerts": alerts}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
