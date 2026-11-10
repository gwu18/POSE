"""Participant-level train/val/test split (no participant appears in two splits).

    python scripts/make_splits.py --val 2 --test 3
    python scripts/make_splits.py --test-pids p03 p07 p11 --val 2
    python scripts/make_splits.py --config configs/synthetic.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posetcn.config import ROOT, load_config, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--val", type=int, default=2)
    ap.add_argument("--test", type=int, default=3)
    ap.add_argument("--test-pids", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    lab_dir = resolve(cfg["paths"]["labels_dir"])
    pids = sorted({p.stem.split("_")[0] for p in resolve(cfg["paths"]["landmarks_dir"]).glob("*.npz")
                   if (lab_dir / p.name.replace(".npz", ".csv")).exists()})
    if len(pids) < 3:
        raise SystemExit(f"need >= 3 labeled participants, found {pids}")
    rng = random.Random(args.seed)
    rest = [p for p in pids if p not in (args.test_pids or [])]
    rng.shuffle(rest)
    test = args.test_pids if args.test_pids is not None else rest[:args.test]
    rest = [p for p in rest if p not in test]
    val, train = rest[:args.val], rest[args.val:]
    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    out = resolve(cfg["paths"]["splits"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2))
    print(f"{out}: " + ", ".join(f"{k} {len(v)} {v}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
