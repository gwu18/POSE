"""Run a small, reproducible transfer-learning hyperparameter sweep.

The held-out test participant is excluded from every sweep run. The validation
participant is also fixed across the sweep so the comparison is fair. After
choosing settings, retrain on all non-test participants and evaluate once on
the held-out participant.

Example once p03 exists:
    python scripts/run_transfer_sweep.py --test-pid p03 --val-pid p02
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

# Allow `python scripts/run_transfer_sweep.py` from the repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from posetcn.config import ROOT, load_config, resolve


TRIALS = [
    {"lr": 0.0001, "dropout": 0.05, "window": 30, "augment": True, "weight_decay": 1e-4},
    {"lr": 0.0003, "dropout": 0.05, "window": 30, "augment": True, "weight_decay": 1e-4},
    {"lr": 0.0010, "dropout": 0.10, "window": 30, "augment": True, "weight_decay": 1e-4},
    {"lr": 0.0003, "dropout": 0.10, "window": 45, "augment": True, "weight_decay": 1e-4},
    {"lr": 0.0003, "dropout": 0.20, "window": 45, "augment": True, "weight_decay": 1e-4},
    {"lr": 0.0003, "dropout": 0.10, "window": 60, "augment": True, "weight_decay": 1e-4},
    {"lr": 0.0003, "dropout": 0.10, "window": 30, "augment": False, "weight_decay": 1e-4},
    {"lr": 0.0003, "dropout": 0.10, "window": 30, "augment": True, "weight_decay": 0.00001},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-pid", default=None, help="single held-out participant (legacy form)")
    ap.add_argument("--test-pids", default=None, help="comma-separated held-out participants")
    ap.add_argument("--val-pid", required=True)
    ap.add_argument("--participants", default=None,
                    help="comma-separated participant pool to use (excludes unrelated labeled clips)")
    ap.add_argument("--init-backbone", default="runs/multiposture/upper_pretrain/best.pt")
    ap.add_argument("--config", default=str(ROOT / "configs/transfer.yaml"))
    ap.add_argument("--name-prefix", default="transfer_sweep_",
                    help="prefix for output run directories")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.test_pids:
        test_pids = [p.strip() for p in args.test_pids.split(",") if p.strip()]
    elif args.test_pid:
        test_pids = [args.test_pid]
    else:
        raise SystemExit("provide --test-pid or --test-pids")

    cfg = load_config(args.config)
    lm_dir, labels_dir = resolve(cfg["paths"]["landmarks_dir"]), resolve(cfg["paths"]["labels_dir"])
    pids = sorted({p.stem.split("_")[0] for p in lm_dir.glob("*.npz")
                   if (labels_dir / f"{p.stem}.csv").exists()})
    if args.participants:
        allowed = {p.strip() for p in args.participants.split(",") if p.strip()}
        pids = [p for p in pids if p in allowed]
    missing = [p for p in [*test_pids, args.val_pid] if p not in pids]
    if missing:
        raise SystemExit(f"test/val pid missing; available labeled participants: {pids}")
    if args.val_pid in test_pids:
        raise SystemExit("validation participant must differ from test participants")
    train = [p for p in pids if p not in {*test_pids, args.val_pid}]
    if not train:
        raise SystemExit("need at least one training participant besides val and test")
    if not Path(args.init_backbone).exists():
        raise SystemExit(f"missing pretrained checkpoint: {args.init_backbone}")

    split_path = resolve(cfg["paths"]["splits"])
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({"train": train, "val": [args.val_pid], "test": test_pids}, indent=2))
    print(f"split: train={train}, val={[args.val_pid]}, test={test_pids}")

    for i, trial in enumerate(TRIALS, 1):
        name = f"{args.name_prefix}{i:02d}"
        overrides = [
            f"train.lr={trial['lr']}",
            f"model.dropout={trial['dropout']}",
            f"data.window={trial['window']}",
            f"augment.enabled={str(trial['augment']).lower()}",
            f"train.weight_decay={trial['weight_decay']:.8f}",
        ]
        cmd = [sys.executable, "-m", "posetcn.train", "--config", args.config,
               "--init-backbone", args.init_backbone, "--name", name, "--set", *overrides]
        print("\n", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
