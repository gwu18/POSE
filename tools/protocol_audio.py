"""Generate a spoken recording protocol (WAV) + matching draft label CSV.

    python tools/protocol_audio.py --pid p01 --session s1_left
    python tools/protocol_audio.py --pid p01 --session s1_left --short   # 3-minute pilot version

Recording procedure
  1. Start the phone VIDEO recording first.
  2. Play data/protocols/p01_s1_left.wav (laptop/speaker). At the SYNC beep the participant claps once.
  3. Save the video as data/raw/p01/s1_left.mp4 (the session name must match).
  4. In tools/label_tool.py, put the cursor on the clap frame and press 'c'. That shifts the draft labels into
     video time. Then fix any boundaries by eye.

Block order is shuffled per session (seeded by pid+session) so the model can't learn the sequence.
Uses the macOS `say` command for speech; elsewhere it writes beeps plus a printed cue sheet for a reader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from posetcn.config import load_config, resolve  # noqa: E402
from posetcn.labels import write_intervals  # noqa: E402

SR = 22050
CUES = {
    "good": "Sit all the way back in the chair. Back supported, shoulders relaxed, ears over your shoulders. Hold it.",
    "forward_head": "Keep your back against the chair, but push your head and chin forward toward the screen.",
    "rounded_shoulders": "Keep your head level, and let your shoulders roll forward and inward, like hugging yourself.",
    "slouched": "Slide your hips forward and let your lower back round off the chair. Slump down.",
    "perched_edge": "Scoot to the front edge of the seat. No back support. Sit there, feet tucked under you.",
    "fh_rs": "Head forward and shoulders rolled in, like leaning in to read the screen.",
    "full_slump": "Full slump. Hips forward, back rounded, shoulders in, head toward your phone.",
    "perched_fh": "Perch on the front edge of the seat and push your head toward the laptop.",
    "transient": "Move around naturally. Take a drink, reach for something, adjust your seat. Keep moving.",
    "free": "Now just sit and work naturally, however you like.",
}
# (cue key, label tag(s) written to CSV, seconds)
# Isolated faults first-class (so each output is learnable on its own), then realistic combinations.
FULL = [("good", "good", 75), ("forward_head", "forward_head", 60), ("rounded_shoulders", "rounded_shoulders", 60),
        ("slouched", "slouched", 60), ("perched_edge", "perched_edge", 60), ("good", "good", 45),
        ("fh_rs", "forward_head+rounded_shoulders", 60),
        ("full_slump", "forward_head+rounded_shoulders+slouched", 60),
        ("perched_fh", "forward_head+perched_edge", 60), ("transient", "transient", 60)]
SHORT = [("good", "good", 30), ("forward_head", "forward_head", 30), ("rounded_shoulders", "rounded_shoulders", 30),
         ("slouched", "slouched", 30), ("perched_edge", "perched_edge", 30), ("transient", "transient", 30)]
FREE = ("free", "free", 75)  # 'free' is not a trainable tag -> ignored until you relabel it by hand
MARGIN_S = 3.0               # unlabeled settling time after each cue


def tone(freq=880, dur=0.25):
    t = np.arange(int(SR * dur)) / SR
    return (0.4 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))).astype(np.float32)


def silence(dur):
    return np.zeros(int(SR * max(dur, 0)), np.float32)


def speak(text: str) -> np.ndarray:
    if not shutil.which("say"):
        return silence(0)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cue.wav"
        subprocess.run(["say", "-o", str(p), f"--data-format=LEI16@{SR}", text], check=True)
        with wave.open(str(p)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing label CSV")
    args = ap.parse_args()

    cfg = load_config()
    name = f"{args.pid}_{args.session}"
    rng = random.Random(int(hashlib.md5(name.encode()).hexdigest(), 16))
    blocks = list(SHORT if args.short else FULL)
    middle = blocks[1:]
    rng.shuffle(middle)
    blocks = [blocks[0]] + middle + ([] if args.short else [FREE])

    audio = [speak("Get ready. Clap once on the beep. Three. Two. One."), tone(1320, 0.4)]
    sync_s = sum(len(a) for a in audio[:1]) / SR
    t = sum(len(a) for a in audio) / SR + 1.0
    audio.append(silence(1.0))
    intervals, sheet = [], [f"{0:7.1f}s  sync beep at {sync_s:.1f}s (clap)"]
    for cue, label, dur in blocks:
        speech = np.concatenate([tone(), speak(CUES[cue])])
        sdur = len(speech) / SR
        audio += [speech, silence(dur - sdur)]
        if label != "free":
            intervals.append((t + sdur + MARGIN_S, t + dur - 0.5, label))
        sheet.append(f"{t:7.1f}s  [{label:<40}] {dur:3d}s  {CUES[cue]}")
        t += max(dur, sdur)
    audio += [tone(660), speak("Session complete. You can stop recording.")]

    out_dir = resolve("data/protocols")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = np.clip(np.concatenate(audio), -1, 1)
    with wave.open(str(out_dir / f"{name}.wav"), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((wav * 32767).astype(np.int16).tobytes())
    (out_dir / f"{name}.txt").write_text("\n".join(sheet) + "\n")
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump({"sync_s": sync_s, "blocks": blocks, "intervals_audio_time": intervals}, f, indent=2)

    label_path = resolve(cfg["paths"]["labels_dir"]) / f"{name}.csv"
    if label_path.exists() and not args.force:
        print(f"kept existing {label_path} (use --force to overwrite)")
    else:
        write_intervals(label_path, intervals)
        print(f"draft labels -> {label_path}  (audio time; press 'c' on the clap in label_tool)")
    if not shutil.which("say"):
        print("NOTE: no `say` command; the WAV has beeps only. Have someone read the cue sheet aloud.")
    print(f"audio -> {out_dir / (name + '.wav')}  ({len(wav) / SR / 60:.1f} min)\n")
    print("\n".join(sheet))


if __name__ == "__main__":
    main()
