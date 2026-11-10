# P.O.S.E. — Posture Optimizer & Support Engine

A side-view phone camera feeds **MediaPipe Pose** (Model C), which produces 33 landmarks. Those go to the **TCN** (Model A), which classifies posture over a 3-second window. An alert engine then triggers haptics and audio, and **Gemini** (Model B) writes the feedback text.

This repository covers the Model A training pipeline plus the tools for collecting and labeling data.

```
posetcn/            the library
  extract.py        video -> MediaPipe landmarks (.npz, resampled to 15 fps)
  features.py       landmarks -> 42 per-frame features (angles, normalised coords, deltas)  <- port to app
  labels.py         interval CSVs <-> per-frame labels
  dataset.py        sliding windows, participant splits, augmentation
  model.py          causal dilated TCN (~96k params, 61-frame receptive field) + frame-MLP baseline
  train.py          masked multi-label BCE, early stopping on val mean fault F1
  evaluate.py       held-out-participant metrics, confusion, per-person F1, alert simulation
  baselines.py      angle-threshold rules
  alerts.py         probability -> buzz logic (threshold + persistence + cooldown)  <- port to app
  export.py         TorchScript / ONNX / TFLite + model_spec.json
  stream_demo.py    live webcam demo on a laptop
tools/
  protocol_audio.py spoken recording script (WAV) + auto-generated draft labels
  label_tool.py     keyboard labeler with skeleton/angle overlay
  qa_labels.py      flags segments whose angles contradict their label
scripts/            download_models.sh, extract_all.sh, make_splits.py, make_synthetic.py
configs/            default.yaml (real data), synthetic.yaml (smoke test)
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/download_models.sh            # MediaPipe pose_landmarker_{lite,full,heavy}.task
pytest -q                               # unit tests
```

Smoke-test the whole pipeline in about 2 minutes, before you have real data:
```bash
python scripts/make_synthetic.py --participants 12
python scripts/make_splits.py --config configs/synthetic.yaml
python -m posetcn.train --config configs/synthetic.yaml --name smoke
python -m posetcn.evaluate runs/synthetic/smoke
```

## Real-data workflow
```bash
# 1. before each recording: generate the spoken protocol + draft labels
python tools/protocol_audio.py --pid p01 --session s1_left
#    start phone video -> play data/protocols/p01_s1_left.wav -> participant claps at the beep
#    save video as data/raw/p01/s1_left.mp4

# 2. landmarks (cached; re-run only with --force)
python -m posetcn.extract --all

# 3. align + fix labels (press 'c' on the clap frame, then nudge boundaries), then QA
python tools/label_tool.py data/raw/p01/s1_left.mp4
python tools/qa_labels.py

# 4. split by participant, train, evaluate
python scripts/make_splits.py --val 2 --test 3
python -m posetcn.train --name tcn_v1
python -m posetcn.evaluate runs/tcn_v1
python -m posetcn.baselines --split train --sweep            # pick thresholds on train ...
python -m posetcn.baselines --split test --thr forward_head=21 slouched=0.96 ...  # ... report on test
python -m posetcn.train --set model.type=frame_mlp --name frame_mlp   # single-frame baseline

# 5. try it live, then export for the app
python -m posetcn.stream_demo runs/tcn_v1
python -m posetcn.export runs/tcn_v1
```

Ablations are config overrides, for example `--set data.window=30` (2 s), `--set augment.enabled=false`, or `--set model.levels=3`.

---

## Labels (multi-label)

The model has **one independent yes/no output per fault**. Each fault drives its own buzzer, and several can fire at once.

| tag | output | buzzer | what to look for (side view) | expected difficulty |
|---|---|---|---|---|
| `good` | — (all faults off) | — | sitting back, back supported, ears over shoulders | easy |
| `forward_head` | yes | neck | ear well ahead of shoulder | **easy** |
| `slouched` | yes | lower back | hips slid forward, lower back rounded off the chair, torso shorter | medium |
| `perched_edge` | yes | seat | on the front edge of the seat, knee lower than hip, feet tucked under | medium |
| `rounded_shoulders` | yes | shoulders | shoulders rolled forward/in while the head may be level | **hard** (subtle in 2D) |
| `transient` | gate | — | drinking, reaching, adjusting; pauses all alerts | easy |

- **Label CSV:** join co-occurring faults with `+`, for example `forward_head+rounded_shoulders+slouched`. Order doesn't matter.
- **How targets are built:**
  - `good` means every output is a known 0.
  - During `transient`, the fault outputs are *unknown* and left out of the loss.
  - Unlabeled frames and `free` are ignored.
- **Label tool keys:** `g` = good, `1`–`4` toggle faults on and off relative to the current segment, `t` = transient, `x` = unlabeled.
- **Metrics:** per-fault precision, recall, F1 and AUROC, plus a derived "good posture" F1 (all faults below threshold). The headline number is the **mean fault F1**.

**No keypoint annotation is needed.** MediaPipe supplies the skeleton. You label time intervals, and `protocol_audio.py` pre-fills them from the spoken cues. The protocol records **each fault on its own first** (so every output can be learned independently) and then the common combinations.

**QA:** `tools/qa_labels.py` prints the median geometry per tag. After the pilot, check that every fault differs visibly from `good` in at least one column, then calibrate the `qa:` thresholds in the config from those numbers. If `rounded_shoulders` is indistinguishable from `good`, decide early whether to keep it (see below).

**If a fault can't be learned:** if a fault's F1 on held-out participants stays below about 0.6, merge it, for example `rounded_shoulders` into `forward_head` as an "upper body" buzzer. Remove it from `classes` and `alert_zones`, and the tags in your CSVs map automatically once you replace them. Report this; it's a real finding about the limits of side-view 2D pose.

## Data to capture

**Target:** at least 12 participants (stretch: 16–20), each with 2 sessions of about 11 minutes, one per side (left and right). That gives each isolated fault about 25 minutes of footage in total, plus the combinations. Use the full protocol (`protocol_audio.py` without `--short`). Use `--short` for pilots.

**Camera:**
- Phone on a tripod, about 90° to the side, lens at seated chest height, 1.5–2.5 m away.
- Head to knees in frame, landscape, 30 fps, 720p or higher.
- Lock exposure, and don't use portrait mode.

**Vary deliberately across sessions** (log each one in a spreadsheet):
- chair (office, dining, couch, stool)
- activity (laptop, monitor, phone, nothing)
- clothing (fitted vs. hoodie)
- lighting (daylight, indoor, backlit window)
- camera yaw (0°, ±20°) and height
- participant body types

**File naming:** `data/raw/<pid>/<session>.mp4`, for example `data/raw/p04/s2_right.mp4`. The pid is the split unit, so never reuse one for a different person.

**Privacy:** get written consent. Raw video stays on the team drive and is gitignored. Landmarks (`.npz`) and labels are de-identified and fine to share.

## External datasets (supplements, not the core)
- **SitPose** (Jin et al., 2024, arXiv:2412.12216): Kinect v2 skeletons of sitting postures. Request or download it on day 1. Use it to **pretrain**: map Kinect joints (Head, SpineShoulder/ShoulderL/R, HipL/R, KneeL/R, ElbowL/R, WristL/R) to the 7 keypoints, project onto the sagittal plane, and write the same `.npz`/`.csv` format. Report results with and without it.
- **Estrada et al., 2023** (Applied Sciences 13(9):5402): check its data-availability statement, and borrow its angle thresholds for the rule baseline.
- **Kaggle/Roboflow sitting-posture image sets**: single frames, so unusable for the TCN. They're useful for checking MediaPipe robustness and as Gemini fine-tuning images.
- No public dataset matches **side-view phone RGB video with temporal labels**, so your own collection is the main dataset. Say this explicitly in the report.

## Evaluation to report
- mean fault F1 (headline), per-fault P/R/F1/AUROC, and good-posture F1, all on **held-out participants**
- per-participant mean fault F1 (shows variance across people)
- TCN vs. frame-MLP vs. rules (justifies the temporal model)
- false alerts per 10 min and median time-to-alert (`alerts` section of `eval_test.json`)
- on-device FPS and latency (app)

## Porting to the app
`export.py` writes `model_spec.json`. The app must reimplement `features.py` exactly (about 80 lines of math) and `alerts.py`. Feature normalisation is baked into the model. To check the port, feed the same landmark window to both implementations and compare features to about 1e-5.
