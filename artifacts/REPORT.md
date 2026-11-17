# P.O.S.E. local-pair TCN model

## Model

`model_best.pt` is the final gated temporal-convolutional network trained from
the verified local video/label pairs. The model has 8,789 parameters and uses a
45-frame input window at 15 Hz (3 seconds).

The external gated-TCN checkpoint used only as backbone initialization is in
`pretraining/gated_backbone_init.pt`.

## Data used

Training data:

- P01: `s1_right.mp4` with `data/labels/p01_s1_right.csv`
- P02: `s2_right.mp4` with `data/labels/p02_s2_right.csv`

Held-out test data:

- P03: `s1_left.mp4` with `data/labels/p03_s1_left.csv`
- P03: `s2_right.mp4` with `data/labels/p03_s2_right.csv`

P03 was not used for hyperparameter selection or training. P02 `s1_left` was
not used because only its CSV/protocol exists locally; its video and landmarks
are missing. P01 sessions without a corresponding local video were excluded.
The videos and label CSVs are not duplicated in this bundle. Raw video is kept
local/private, and the matching labels remain in the repository's main
`data/labels/` directory.

## Hyperparameter search

All trials used the gated TCN architecture, 16 channels, 3 levels, kernel size
3, the same gated pretrained backbone, and P01 training → P02 validation.
P03 remained held out.

| Trial | Learning rate | Dropout | Window | Augmentation | Weight decay | P02 mean fault F1 |
|---:|---:|---:|---:|:---:|---:|---:|
| 1 | 0.0001 | 0.05 | 30 | yes | 0.0001 | 0.680 |
| 2 | 0.0003 | 0.05 | 30 | yes | 0.0001 | 0.677 |
| 3 | 0.0010 | 0.10 | 30 | yes | 0.0001 | 0.679 |
| 4 | 0.0003 | 0.10 | 45 | yes | 0.0001 | 0.708 |
| **5** | **0.0003** | **0.20** | **45** | **yes** | **0.0001** | **0.710** |
| 6 | 0.0003 | 0.10 | 60 | yes | 0.0001 | 0.680 |
| 7 | 0.0003 | 0.10 | 30 | no | 0.0001 | 0.627 |
| 8 | 0.0003 | 0.10 | 30 | yes | 0.00001 | 0.672 |

Trial 5 was selected using P02 validation mean fault F1. Its settings were
then used to retrain on both P01 and P02 for 12 epochs before the final P03
evaluation.

## Results

Final training-set diagnostics on P01+P02:

- Mean fault F1: 0.684
- Exact-match accuracy: 0.295
- Good-posture F1: 0.588

Final held-out P03 test results:

- Mean fault F1: **0.343**
- Exact-match accuracy: **0.124**
- Good-posture F1: 0.237
- False alerts: 8
- Missed fault segments: 17
- False alerts per 10 minutes: 4.67

The full machine-readable test report is `eval_test.json`; training metrics are
in `final_training_metrics.json`.

## Reproduction code

The source snapshot used by the training/evaluation pipeline is under
`code/posetcn/`. The sweep driver is under `code/scripts/`. The effective
configuration and participant split are `config.yaml` and `splits.json`.
