"""Live webcam demo on a laptop: MediaPipe -> rolling window -> TCN -> alerts.

    python -m posetcn.stream_demo runs/<name>              # default webcam
    python -m posetcn.stream_demo runs/<name> --source video.mp4

This is the reference implementation of what the mobile app does each frame.
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import torch

from .alerts import AlertEngine
from .config import resolve
from .extract import make_landmarker, result_to_arrays
from .features import canonical_points, choose_side, geometry, points_to_features, torso_length
from .model import load_run



def main(argv=None):
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--source", default="0")
    args = ap.parse_args(argv)

    model, cfg, _ = load_run(args.run_dir)
    classes, W, fps = cfg["classes"], cfg["data"]["window"], cfg["data"]["fps"]
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    mp, landmarker = make_landmarker(resolve(cfg["paths"]["pose_model"]))
    engine = AlertEngine(cfg)

    raw = deque(maxlen=W)                 # raw (33,4) landmarks at `fps`
    torsos = deque(maxlen=int(60 * fps))  # for the running reference scale
    side = facing = None
    probs = np.zeros(len(classes))
    t0, next_t, last_alerts = time.time(), 0.0, []
    with landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time() - t0
            H, Wd = frame.shape[:2]
            if t >= next_t:
                next_t = t + 1.0 / fps
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    int(t * 1000))
                lm, _ = result_to_arrays(res)
                raw.append(lm)
                arr = np.stack(raw)
                if side is None and np.isfinite(arr[:, 0, 0]).sum() >= min(2 * fps, W):
                    side = choose_side(arr)
                    _, _, _, meta = canonical_points(arr, Wd, H, side=side)
                    facing = meta["facing"]
                if side is not None and len(raw) == W:
                    pts, vis, det, _ = canonical_points(arr, Wd, H, side=side, facing=facing)
                    if det[-1]:
                        torsos.append(float(torso_length(pts[-1])))
                    ref = float(np.median(torsos)) if torsos else None
                    x = torch.from_numpy(points_to_features(pts, vis, det, ref))[None]
                    with torch.no_grad():
                        probs = torch.sigmoid(model(x))[0].numpy()
                    alerts = engine.update(t, probs)
                    if alerts:
                        last_alerts = alerts
                        for al in alerts:
                            print(f"[{t:6.1f}s] BUZZ {al.zone}: {al.cls}")
                    g = geometry(pts[-1:], ref)[0]
                    cv2.putText(frame, f"neck {g[0] - g[1]:5.1f}  trunk {g[1]:5.1f}  thigh {g[8]:5.1f}  "
                                f"sh-line {g[10]:+.2f}", (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 2)
                if np.isfinite(lm[0, 0]):
                    for p in lm[[0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28]]:
                        cv2.circle(frame, (int(p[0] * Wd), int(p[1] * H)), 4, (255, 255, 0), -1)

            # one probability bar per output; "GOOD" when every fault is below threshold
            faults = [classes.index(c) for c in cfg["alert_zones"]]
            good = all(probs[i] < engine.thr[i] for i in faults)
            cv2.putText(frame, "GOOD POSTURE" if good else "FIX:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (80, 200, 80) if good else (0, 0, 255), 2)
            for r, c in enumerate(classes):
                y = 60 + 26 * r
                p = float(probs[r])
                col = (0, 0, 255) if p >= engine.thr.get(r, cfg["alerts"]["gate_threshold"]) else (180, 180, 180)
                cv2.rectangle(frame, (10, y - 14), (10 + int(150 * p), y), col, -1)
                cv2.putText(frame, f"{c} {p:.2f}", (170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            for al in last_alerts:
                if t - al.t < 3:
                    cv2.putText(frame, f"BUZZ {al.zone.upper()}", (Wd - 300, 40 + 35 * last_alerts.index(al)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow("P.O.S.E. demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
