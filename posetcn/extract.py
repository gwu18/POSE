"""Video -> cached MediaPipe Pose landmarks (.npz), resampled to cfg.data.fps.

    python -m posetcn.extract data/raw/p01/s1_left.mp4          # one video
    python -m posetcn.extract --all                              # every data/raw/<pid>/*.mp4

Output data/landmarks/<pid>_<session>.npz with:
    lm     (T,33,4) normalised image x,y,z,visibility  (NaN rows = no person detected)
    world  (T,33,3) metric world landmarks (hip-centred)
    t      (T,)     timestamps in seconds
    fps, width, height, pid, session, video
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import ROOT, load_config, resolve


def make_landmarker(model_path: Path):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not model_path.exists():
        raise SystemExit(f"missing {model_path}; run scripts/download_models.sh")
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, vision.PoseLandmarker.create_from_options(opts)


def result_to_arrays(res) -> tuple[np.ndarray, np.ndarray]:
    lm = np.full((33, 4), np.nan, dtype=np.float32)
    world = np.full((33, 3), np.nan, dtype=np.float32)
    if res.pose_landmarks:
        lm[:] = [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]]
        world[:] = [[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]]
    return lm, world


def extract_video(video: Path, out: Path, model_path: Path, fps: float) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mp, landmarker = make_landmarker(model_path)
    lms, worlds, ts = [], [], []
    next_t, k, W, H = 0.0, 0, 0, 0
    with landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = k / src_fps
            k += 1
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / fps
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            lm, world = result_to_arrays(res)
            lms.append(lm)
            worlds.append(world)
            ts.append(t)
    cap.release()

    pid = video.parent.name
    session = video.stem
    out.parent.mkdir(parents=True, exist_ok=True)
    lm_arr = np.stack(lms) if lms else np.zeros((0, 33, 4), np.float32)
    np.savez_compressed(out, lm=lm_arr, world=np.stack(worlds) if worlds else np.zeros((0, 33, 3), np.float32),
                        t=np.array(ts, dtype=np.float64), fps=fps, width=W, height=H,
                        pid=pid, session=session, video=str(video))
    det = np.isfinite(lm_arr[:, 0, 0]).mean() if len(lm_arr) else 0.0
    return {"frames": len(ts), "detected": float(det), "size": (W, H)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="process data/raw/<pid>/*.{mp4,mov}")
    ap.add_argument("--force", action="store_true", help="re-extract even if output exists")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    videos = list(args.videos)
    if args.all:
        raw = resolve(cfg["paths"]["raw_dir"])
        videos += sorted(p for p in raw.glob("*/*") if p.suffix.lower() in {".mp4", ".mov", ".m4v"})
    if not videos:
        ap.error("give video paths or --all")

    out_dir = resolve(cfg["paths"]["landmarks_dir"])
    model_path = resolve(cfg["paths"]["pose_model"])
    for v in videos:
        out = out_dir / f"{v.parent.name}_{v.stem}.npz"
        if out.exists() and not args.force:
            print(f"skip {out.name} (exists)")
            continue
        info = extract_video(v, out, model_path, cfg["data"]["fps"])
        flag = "  <-- LOW DETECTION, check framing/lighting" if info["detected"] < 0.8 else ""
        print(f"{out.name}: {info['frames']} frames @ {cfg['data']['fps']} fps, {info['size']}, "
              f"detected {info['detected']:.0%}{flag}")


if __name__ == "__main__":
    main()
