# Agent context — P.O.S.E.

Read this for project orientation. Human-facing setup and workflows live in `README.md`.

## What this project is

**P.O.S.E. (Posture Optimizer & Support Engine)** — an assistive sitting-posture system. Side-view phone video → MediaPipe Pose → geometric features → TCN over a ~3s window → per-fault probabilities → haptic/audio alerts (Gemini later for spoken/text feedback).

**This repository** is the Model A (TCN) training pipeline plus data collection/labeling tools. It is **not** the mobile app, Gemini fine-tuning, or hardware buzzers.

## Pipeline

```
Camera (side view, phone)
  → Model C: MediaPipe Pose Landmarker (33 landmarks)
  → features.py: 42 per-frame features (angles, normalized coords, deltas)
  → Model A: Causal dilated TCN (~96k params, 45-frame / 3s window @ 15 fps)
  → AlertEngine: threshold + persistence + cooldown per fault
  → Model B: Gemini (feedback text; not implemented in this repo)
```

## Repo layout

| Path | Role |
|---|---|
| `posetcn/` | Library: extract, features, labels, dataset, model, train, evaluate, baselines, alerts, export, stream_demo |
| `tools/` | `protocol_audio.py` (spoken script + draft labels), `label_tool.py`, `qa_labels.py` |
| `scripts/` | download MediaPipe models, extract_all, make_splits, make_synthetic |
| `configs/` | `default.yaml` (real data), `synthetic.yaml` (smoke test) |
| `data/raw/` | **Where videos go** (gitignored) |
| `data/landmarks/` | Cached MediaPipe `.npz` (ok to share) |
| `data/labels/` | Interval CSVs (ok to share) |
| `data/protocols/` | Spoken protocol JSON/TXT/WAV for recording sessions |
| `models/` | MediaPipe `.task` files (gitignored; download via script) |
| `runs/` | Training runs + exports (gitignored) |

## Labels (multi-label)

One independent yes/no output per fault; several can fire at once. `good` = all faults off.

| Tag | Buzzer zone | Notes |
|---|---|---|
| `forward_head` | neck | Ear ahead of shoulder |
| `rounded_shoulders` | shoulders | Hard in 2D side view |
| `slouched` | lower_back | Hips forward, rounded back |
| `perched_edge` | seat | Front edge of seat |
| `transient` | gate | Drinking/reaching — pauses alerts; faults masked in loss |
| `good` | — | All fault outputs known 0 |

CSV format: `start_s,end_s,label` with `+`-joined co-occurring faults.

## Where to put videos

```
data/raw/<pid>/<session>.mp4
```

Examples:

- `data/raw/p01/s1_left.mp4`
- `data/raw/p01/s1_right.mp4`
- `data/raw/p02/s2_right.mp4`

Rules:

- `pid` is the **participant id** and the train/val/test split unit — never reuse a pid for a different person.
- Session names are typically `s1_left`, `s1_right`, `s2_left`, `s2_right`.
- Raw video is **gitignored** (`data/raw/`) for privacy; keep originals on the team drive.
- Matching labels go in `data/labels/<pid>_<session>.csv` (e.g. `p01_s1_left.csv`).
- After placing a video, extract landmarks: `python -m posetcn.extract --all` (or a single path).

## Typical real-data workflow

1. `python tools/protocol_audio.py --pid p01 --session s1_left` → play WAV while recording; clap at beep.
2. Save video as `data/raw/p01/s1_left.mp4`.
3. `python -m posetcn.extract --all`
4. `python tools/label_tool.py data/raw/p01/s1_left.mp4` then `python tools/qa_labels.py`
5. `python scripts/make_splits.py --val 2 --test 3`
6. `python -m posetcn.train --name tcn_v1` → `python -m posetcn.evaluate runs/tcn_v1`
7. Live check: `python -m posetcn.stream_demo runs/tcn_v1`
8. App export: `python -m posetcn.export runs/tcn_v1` (app must port `features.py` + `alerts.py` exactly)

## Proposal vs this codebase

**Aligned:** MediaPipe → TCN → alerts; side-view RGB; SitPose optional pretrain; participant-held-out eval; export for mobile.

**Sharper here than the short proposal:** multi-label faults + transient gate; persistence/cooldown alerts; rule + frame-MLP ablations; headline metric = mean fault F1.

**Not in this repo:** Gemini fine-tuning, mobile UI, physical buzzers, session report UI.

## Data collection status (as of last write)

- Protocols exist for early p01/p02 sessions under `data/protocols/`.
- Some draft/aligned label CSVs under `data/labels/`.
- `data/raw/` and `data/landmarks/` may be empty in a fresh clone — videos are local/private.
- Synthetic smoke path: `scripts/make_synthetic.py` + `configs/synthetic.yaml`.
- Target: ≥12 participants × 2 side sessions (~11 min), camera ~90° side, chest height, 1.5–2.5 m, landscape 30 fps.

## Porting notes for the app

- `export.py` writes `model_spec.json` + ONNX/TorchScript/(optional) TFLite.
- Feature normalization is baked into the model.
- Reimplement `features.py` and `alerts.py` on device; compare features to ~1e-5 vs Python for a golden window.
