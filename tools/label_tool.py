"""Keyboard-driven interval labeler with skeleton + angle overlay.

    python tools/label_tool.py data/raw/p01/s1_left.mp4

Loads the draft labels from protocol_audio.py if present (data/labels/<pid>_<session>.csv),
so you mostly just nudge boundaries.

Keys
  space        play / pause
  a / d        back / forward 1 s       (A / D = 0.2 s)
  j / l        back / forward 10 s
  g            a "good" segment starts HERE
  1-4          a new segment starts HERE with that fault TOGGLED relative to the current segment
               (1=forward_head 2=rounded_shoulders 3=slouched 4=perched_edge; combine freely)
  t            a "transient" segment starts here (drinking, reaching, adjusting)
  x            "unlabeled" starts here (transitions, free sitting you skip, junk)
  r            remove the boundary nearest the cursor
  c            the participant's clap is HERE: aligns protocol draft labels to video time
  [ / ]        shift ALL boundaries -0.5 s / +0.5 s (fine sync)
  s            save        q   save + quit       Esc  quit without saving
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from posetcn.config import load_config, resolve  # noqa: E402
from posetcn.features import canonical_points, geometry, reference_scale  # noqa: E402
from posetcn.labels import normalize_tag, read_intervals, write_intervals  # noqa: E402

FAULT_COLORS = {"forward_head": (0, 140, 255), "rounded_shoulders": (200, 0, 200), "slouched": (0, 0, 255),
                "perched_edge": (0, 220, 220)}


def tag_color(tag):
    if not tag:
        return (90, 90, 90)
    if tag == "good":
        return (80, 200, 80)
    if tag == "transient":
        return (200, 200, 200)
    parts = [p for p in tag.split("+") if p in FAULT_COLORS]
    if not parts:
        return (150, 150, 150)
    return tuple(int(np.mean([FAULT_COLORS[p][k] for p in parts])) for k in range(3))
SKELETON = [(7, 11), (11, 23), (23, 25), (25, 27), (11, 13), (13, 15), (0, 7),
            (8, 12), (12, 24), (24, 26), (26, 28), (12, 14), (14, 16), (0, 8)]


def intervals_to_bounds(iv):
    b = {}
    for s, e, lab in iv:
        b[round(s, 2)] = lab
        b.setdefault(round(e, 2), None)
    return b


def bounds_to_intervals(b, duration):
    keys = sorted(b)
    out = []
    for i, k in enumerate(keys):
        end = keys[i + 1] if i + 1 < len(keys) else duration
        if b[k] is not None and end > k:
            out.append((k, end, b[k]))
    return out


def label_at(b, t):
    ks = [k for k in sorted(b) if k <= t]
    return b[ks[-1]] if ks else None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    video = Path(sys.argv[1])
    cfg = load_config()
    gate = cfg.get("gate_class")
    faults = [c for c in cfg["classes"] if c != gate]
    name = f"{video.parent.name}_{video.stem}"
    label_path = resolve(cfg["paths"]["labels_dir"]) / f"{name}.csv"
    npz_path = resolve(cfg["paths"]["landmarks_dir"]) / f"{name}.npz"

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps

    lm = geo = lt = None
    if npz_path.exists():
        d = np.load(npz_path)
        lm, lt = d["lm"], d["t"]
        pts, _, det, _ = canonical_points(lm, int(d["width"]), int(d["height"]))
        geo = geometry(pts, reference_scale(pts, det))
    else:
        print(f"(no landmarks at {npz_path}; run extraction for skeleton overlay)")

    bounds = intervals_to_bounds(read_intervals(label_path)) if label_path.exists() else {}
    proto = resolve("data/protocols") / f"{name}.json"
    sync_s = json.loads(proto.read_text())["sync_s"] if proto.exists() else None
    shifted = 0.0
    print("faults:", ", ".join(f"{i + 1}={c}" for i, c in enumerate(faults)))
    print(__doc__.split("Keys")[1])

    pos, playing, dirty = 0, False, False
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            pos, playing = max(0, n_frames - 1), False
            continue
        t = pos / fps
        H, W = frame.shape[:2]

        if lm is not None:
            k = int(np.argmin(np.abs(lt - t)))
            p = lm[k]
            if np.isfinite(p[0, 0]):
                for a, b in SKELETON:
                    if min(p[a, 3], p[b, 3]) > 0.3:
                        cv2.line(frame, (int(p[a, 0] * W), int(p[a, 1] * H)), (int(p[b, 0] * W), int(p[b, 1] * H)),
                                 (255, 255, 0), 2)
                g = geo[k]
                cv2.putText(frame, f"neck {g[0]:5.1f}  trunk {g[1]:5.1f}  hip {g[2]:5.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cur = label_at(bounds, t)
        cv2.putText(frame, f"{t:7.2f}s  [{cur or 'unlabeled'}]{'  *unsaved*' if dirty else ''}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tag_color(cur), 2)

        # timeline
        y0 = H - 24
        cv2.rectangle(frame, (0, y0), (W, H), (40, 40, 40), -1)
        for s, e, lab in bounds_to_intervals(bounds, duration):
            c = tag_color(lab)
            cv2.rectangle(frame, (int(s / duration * W), y0), (int(e / duration * W), H), c, -1)
        cx = int(t / duration * W)
        cv2.line(frame, (cx, y0 - 6), (cx, H), (255, 255, 255), 2)

        cv2.imshow("label_tool", frame)
        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF
        step = {ord("a"): -1, ord("d"): 1, ord("A"): -0.2, ord("D"): 0.2, ord("j"): -10, ord("l"): 10}
        if key == ord(" "):
            playing = not playing
        elif key in step:
            pos = int(np.clip(pos + step[key] * fps, 0, n_frames - 1))
            continue
        elif ord("1") <= key <= ord("9") and key - ord("1") < len(faults):
            f = faults[key - ord("1")]
            cur_parts = set((cur or "").split("+")) - {"", "good", "transient"}
            cur_parts ^= {f}
            bounds[round(t, 2)] = normalize_tag("+".join(cur_parts)) if cur_parts else "good"
            dirty = True
        elif key == ord("g"):
            bounds[round(t, 2)] = "good"
            dirty = True
        elif key == ord("t"):
            bounds[round(t, 2)] = "transient"
            dirty = True
        elif key == ord("x"):
            bounds[round(t, 2)] = None
            dirty = True
        elif key == ord("r") and bounds:
            del bounds[min(bounds, key=lambda k: abs(k - t))]
            dirty = True
        elif key == ord("c"):
            if sync_s is None:
                print(f"no protocol file {proto}; use [ and ] instead")
            else:
                dt = (t - sync_s) - shifted
                bounds = {round(max(0.0, k + dt), 2): v for k, v in bounds.items()}
                shifted += dt
                dirty = True
                print(f"shifted labels by {shifted:+.2f}s total")
        elif key in (ord("["), ord("]")):
            dt = -0.5 if key == ord("[") else 0.5
            bounds = {round(max(0.0, k + dt), 2): v for k, v in bounds.items()}
            dirty = True
        elif key in (ord("s"), ord("q")):
            write_intervals(label_path, bounds_to_intervals(bounds, duration))
            print(f"saved {label_path}")
            dirty = False
            if key == ord("q"):
                break
        elif key == 27:
            break
        if playing:
            pos = min(pos + 1, n_frames - 1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
