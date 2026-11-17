# Reproduction commands

Run these commands from the repository root with the project virtual
environment active. The raw videos and extracted landmarks must be available in
the repository's local data directories.

The final run used the effective settings in `config.yaml` and the split in
`splits.json`:

```bash
./.venv/bin/python -m posetcn.train \
  --config configs/gated_transfer.yaml \
  --train-only \
  --init-backbone runs/multiposture/archpre_gated16x3/best.pt \
  --name localpair_final_p01_p02 \
  --set \
    paths.landmarks_dir=data/local_pair_landmarks \
    paths.labels_dir=data/local_pair_labels \
    paths.splits=data/local_pair_splits.json \
    train.lr=0.0003 \
    model.dropout=0.2 \
    data.window=45 \
    augment.enabled=true \
    train.weight_decay=0.0001 \
    train.epochs=12

./.venv/bin/python -m posetcn.evaluate \
  runs/localpair_final_p01_p02 --split test
```

The eight-trial selection sweep was run with:

```bash
./.venv/bin/python scripts/run_transfer_sweep.py \
  --test-pid p03 --val-pid p02 \
  --participants p01,p02,p03 \
  --config configs/gated_transfer.yaml \
  --init-backbone runs/multiposture/archpre_gated16x3/best.pt \
  --name-prefix localpair_gated_
```
