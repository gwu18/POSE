"""Flag labeled segments whose geometry contradicts their tags -> review these in label_tool.

    python tools/qa_labels.py [--config configs/default.yaml]

Also prints per-tag median angles, so you can see whether e.g. `rounded_shoulders` actually
differs from `good` in your data. If it doesn't, the model can't learn it from side-view landmarks.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from posetcn.config import ROOT, load_config  # noqa: E402
from posetcn.dataset import load_clips  # noqa: E402
from posetcn.features import geometry  # noqa: E402
from posetcn.labels import frames_to_intervals  # noqa: E402


def check(tags: set[str], m: dict, q: dict) -> list[str]:
    """m: segment medians. Returns problems."""
    out = []
    if tags == {"good"}:
        if m["rel_neck"] > q["good_max_rel_neck"]:
            out.append(f"good but head forward (rel neck {m['rel_neck']:.0f})")
        if abs(m["trunk"]) > q["good_max_abs_trunk"]:
            out.append(f"good but trunk {m['trunk']:.0f}")
    if "forward_head" in tags and m["rel_neck"] < q["forward_head_min_rel_neck"]:
        out.append(f"forward_head but rel neck only {m['rel_neck']:.0f}")
    if "rounded_shoulders" in tags and m["sh_line"] < q["rounded_shoulders_min_line"]:
        out.append(f"rounded_shoulders but shoulder-line {m['sh_line']:+.2f}")
    if "slouched" in tags and not (m["torso_ratio"] < q["slouched_max_torso_ratio"]
                                   or m["trunk"] < q["slouched_max_trunk"]):
        out.append(f"slouched but torso ratio {m['torso_ratio']:.2f}, trunk {m['trunk']:.0f}")
    if "perched_edge" in tags and m["thigh"] < q["perched_edge_min_thigh"]:
        out.append(f"perched_edge but thigh only {m['thigh']:.0f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    cfg = load_config(ap.parse_args().config)
    q = cfg["qa"]
    flagged = total = 0
    stats = defaultdict(list)
    for c in load_clips(cfg):
        g = geometry(c.pts, c.ref)
        for s, e, tag in frames_to_intervals(c.tags, c.t):
            seg = (c.t >= s) & (c.t < e)
            m = seg & (c.det > 0)
            if m.sum() < 5:
                continue
            total += 1
            med = {"rel_neck": float(np.median(g[m, 0] - g[m, 1])), "trunk": float(np.median(g[m, 1])),
                   "torso_ratio": float(np.median(g[m, 7])), "thigh": float(np.median(g[m, 8])),
                   "sh_line": float(np.median(g[m, 10]))}
            stats[tag].append(med)
            problems = check(set(tag.split("+")), med, q) if tag != "transient" else []
            det = c.det[seg].mean()
            if det < 0.8:
                problems.append(f"pose detected only {det:.0%}")
            if problems:
                flagged += 1
                print(f"{c.name:<24}{s:8.1f}-{e:<8.1f}{tag:<40}{'; '.join(problems)}")
    print(f"\n{flagged}/{total} segments flagged\n\nmedian geometry per tag:")
    print(f"  {'tag':<44}{'relneck':>8}{'trunk':>7}{'torso':>7}{'thigh':>7}{'shline':>8}{'n':>5}")
    for tag in sorted(stats):
        a = {k: np.median([d[k] for d in stats[tag]]) for k in stats[tag][0]}
        print(f"  {tag:<44}{a['rel_neck']:8.1f}{a['trunk']:7.1f}{a['torso_ratio']:7.2f}{a['thigh']:7.1f}"
              f"{a['sh_line']:+8.2f}{len(stats[tag]):5d}")


if __name__ == "__main__":
    main()
