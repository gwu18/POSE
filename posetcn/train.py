"""Train the posture TCN (or the frame-MLP baseline).

    python -m posetcn.train --config configs/default.yaml
    python -m posetcn.train --set model.type=frame_mlp --name mlp_baseline
    python -m posetcn.train --overfit-small        # pipeline sanity check
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from .config import ROOT, load_config, resolve
from .dataset import WindowDataset, feature_stats, load_clips, load_splits
from .features import NUM_FEATURES
from .metrics import format_report, report
from .model import build_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device):
    """-> Y, P (sigmoid probabilities), M as numpy (N,C)."""
    model.eval()
    ys, ps, ms = [], [], []
    for x, y, m in loader:
        ps.append(torch.sigmoid(model(x.to(device))).cpu())
        ys.append(y)
        ms.append(m)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(ms).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=20 model.type=frame_mlp")
    ap.add_argument("--name", default=None, help="run name (default: timestamp)")
    ap.add_argument("--overfit-small", action="store_true", help="train on 256 windows, no aug, expect F1 ~1.0")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    tc = cfg["train"]
    seed_everything(tc["seed"])
    device = pick_device()
    classes = cfg["classes"]

    splits = load_splits(cfg)
    train_clips = load_clips(cfg, splits["train"])
    val_clips = load_clips(cfg, splits.get("val") or [])
    if not train_clips:
        raise SystemExit("no training clips found; run extraction + labeling first (see README)")
    print(f"clips: train {len(train_clips)} ({len(splits['train'])} pids), val {len(val_clips)}")

    train_ds = WindowDataset(train_clips, cfg, augment=not args.overfit_small)
    if args.overfit_small:
        idx = np.random.default_rng(0).choice(len(train_ds), size=min(256, len(train_ds)), replace=False)
        train_view, val_ds = Subset(train_ds, idx), Subset(train_ds, idx)
        tc = {**tc, "epochs": 150, "patience": 10_000}
        cfg["model"]["dropout"] = 0.0
    else:
        if not val_clips:
            raise SystemExit("splits.json has no val participants (needed for early stopping)")
        train_view = train_ds
        val_ds = WindowDataset(val_clips, cfg, augment=False)

    pos, known = train_ds.positives(), train_ds.knowns()
    print("train windows (positive / known) per output:",
          ", ".join(f"{c} {int(p)}/{int(k)}" for c, p, k in zip(classes, pos, known)))
    print(f"train windows {len(train_view)}, val windows {len(val_ds)}")

    model = build_model(cfg, NUM_FEATURES)
    mean, std = feature_stats(train_ds)
    model.set_norm(mean, std)
    model.to(device)
    print(f"model {cfg['model']['type']}: {sum(p.numel() for p in model.parameters()):,} params, "
          f"receptive field {model.receptive_field} frames, device {device}")

    pw = np.ones(len(classes), np.float32)
    if tc["pos_weight"]:
        pw = np.clip((known - pos) / np.maximum(pos, 1), 1.0, tc["pos_weight_max"]).astype(np.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device), reduction="none")

    def loss_fn(logits, y, m):
        return (bce(logits, y) * m).sum() / m.sum().clamp(min=1)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])

    train_dl = DataLoader(train_view, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"],
                          drop_last=len(train_view) > tc["batch_size"])
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    run_dir = resolve(cfg["paths"]["runs_dir"]) / (args.name or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(tc["epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, m in train_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = loss_fn(model(x), y, m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()

        Y, P, M = predict(model, val_dl, device)
        r = report(Y, P, M, cfg)
        history.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_mean_fault_f1": r["mean_fault_f1"],
                        "val_good_f1": r["good"]["f1"]})
        print(f"epoch {epoch:3d}  loss {tot / max(n, 1):.4f}  val fault-F1 {r['mean_fault_f1']:.3f}  "
              f"good-F1 {r['good']['f1']:.3f}  ({time.time() - t0:.1f}s)")

        if r["mean_fault_f1"] > best:
            best, best_epoch = r["mean_fault_f1"], epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg, "num_features": NUM_FEATURES,
                        "classes": classes, "epoch": epoch, "val": r, "splits": splits}, run_dir / "best.pt")
        elif epoch - best_epoch >= tc["patience"]:
            print(f"early stop (best epoch {best_epoch})")
            break

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    print(f"\nbest epoch {ckpt['epoch']}  ({'train subset' if args.overfit_small else 'val'})\n")
    print(format_report(ckpt["val"], classes))
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"best_epoch": ckpt["epoch"], "val": ckpt["val"], "history": history}, f, indent=2)
    print(f"\nrun saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
