"""Rule-based baseline: one angle threshold per fault, no learning. Compare against the TCN.

    python -m posetcn.baselines --split train --sweep     # pick thresholds on train only
    python -m posetcn.baselines --split test --thr forward_head=22 slouched=0.95 ...

The single-frame learned baseline is the frame_mlp model:
    python -m posetcn.train --set model.type=frame_mlp --name frame_mlp
"""
from __future__ import annotations

import argparse

import numpy as np

from .config import ROOT, load_config
from .dataset import load_clips, load_splits
from .features import geometry
from .metrics import format_report, report

# fault -> (signal from geometry g (T,11), direction, default threshold, sweep grid)
RULES = {
    "forward_head": (lambda g: g[:, 0] - g[:, 1], ">", 20.0, np.arange(5, 45, 1.0)),          # rel. neck angle
    "rounded_shoulders": (lambda g: g[:, 10], ">", 0.05, np.arange(-0.1, 0.3, 0.01)),          # shoulder ahead of ear-hip line
    "slouched": (lambda g: g[:, 7], "<", 0.95, np.arange(0.8, 1.05, 0.005)),                   # torso shorter than usual
    "perched_edge": (lambda g: g[:, 8], ">", 10.0, np.arange(-10, 35, 1.0)),                   # knee below hip
}


def rule_probs(g: np.ndarray, classes: list[str], thr: dict) -> np.ndarray:
    """0/1 'probabilities' per output (transient is never predicted by rules)."""
    P = np.zeros((len(g), len(classes)), np.float32)
    for c, (fn, d, _, _) in RULES.items():
        if c in classes:
            v = fn(g)
            P[:, classes.index(c)] = (v > thr[c]) if d == ">" else (v < thr[c])
    return P


def collect(cfg, pids):
    gs, Ys, Ms = [], [], []
    for c in load_clips(cfg, pids):
        k = c.M.max(1) > 0
        gs.append(geometry(c.pts, c.ref)[k])
        Ys.append(c.Y[k])
        Ms.append(c.M[k])
    return np.concatenate(gs), np.concatenate(Ys), np.concatenate(Ms)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr", nargs="*", default=[], help="fault=value overrides")
    ap.add_argument("--sweep", action="store_true", help="choose each threshold to maximise its F1 on this split")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["classes"]
    g, Y, M = collect(cfg, load_splits(cfg)[args.split])
    thr = {c: r[2] for c, r in RULES.items()}
    thr.update({k: float(v) for k, v in (s.split("=") for s in args.thr)})
    if args.sweep:
        for c, (_, _, _, grid) in RULES.items():
            if c not in classes:
                continue
            i = classes.index(c)
            best = max(grid, key=lambda v: report(Y, rule_probs(g, classes, {**thr, c: v}), M, cfg)
                       ["per_class"][c]["f1"])
            thr[c] = float(best)
        print(f"best thresholds on {args.split}: " + " ".join(f"{c}={v:g}" for c, v in thr.items()))
    print(f"rules on {args.split}, per-frame:\n")
    print(format_report(report(Y, rule_probs(g, classes, thr), M, cfg), classes))
    print("\n(rules never predict transient, so that row is zero by design)")


if __name__ == "__main__":
    main()
