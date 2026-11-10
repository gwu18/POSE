"""Interval label files <-> per-frame multi-label targets.

CSV format (data/labels/<pid>_<session>.csv), label = one or more tags joined with '+':
    start_s,end_s,label
    3.0,90.0,good
    93.0,150.0,forward_head
    153.0,210.0,forward_head+rounded_shoulders+slouched
    213.0,270.0,transient
Frames not covered by any interval are unlabeled and never used for training.

Targets per frame: Y (T,C) in {0,1} and mask M (T,C) in {0,1} (1 = known).
  good                    -> all outputs 0, all known
  fault tags (a+b)        -> listed faults 1, other faults 0, transient 0, all known
  transient               -> transient 1; fault outputs UNKNOWN (masked)
  unknown tag (e.g. free) -> unlabeled
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np


def read_intervals(path: str | Path) -> list[tuple[float, float, str]]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append((float(row["start_s"]), float(row["end_s"]), normalize_tag(row["label"])))
    return sorted(out)


def write_intervals(path: str | Path, intervals: list[tuple[float, float, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "label"])
        for s, e, lab in sorted(intervals):
            w.writerow([f"{s:.3f}", f"{e:.3f}", lab])


def normalize_tag(label: str) -> str:
    """Canonical tag string: sorted, de-duplicated, '+'-joined."""
    parts = sorted({p.strip() for p in label.split("+") if p.strip()})
    return "+".join(parts)


def tag_to_target(tag: str, cfg: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """One tag string -> (y, m) vectors of length C, or None if the tag isn't trainable."""
    classes, gate, good = cfg["classes"], cfg.get("gate_class"), cfg.get("good_tag", "good")
    C = len(classes)
    parts = [p for p in tag.split("+") if p]
    if not parts or any(p not in classes and p != good for p in parts):
        return None
    y, m = np.zeros(C, np.float32), np.ones(C, np.float32)
    if parts == [good]:
        return y, m
    if good in parts:
        warnings.warn(f"tag '{tag}' mixes '{good}' with faults; ignoring '{good}'")
    for p in parts:
        if p != good:
            y[classes.index(p)] = 1.0
    if gate and gate in parts:
        for i, c in enumerate(classes):
            if c != gate and c not in parts:
                m[i] = 0.0  # faults unknown during transient movement
    return y, m


def frame_tags(intervals: list[tuple[float, float, str]], t: np.ndarray) -> np.ndarray:
    """Per-frame tag strings ('' = unlabeled)."""
    tags = np.full(len(t), "", dtype=object)
    for s, e, lab in intervals:
        tags[(t >= s) & (t < e)] = lab
    return tags


def frame_targets(
    intervals: list[tuple[float, float, str]], t: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> Y (T,C), M (T,C), tags (T,) for frame timestamps t (seconds)."""
    C = len(cfg["classes"])
    tags = frame_tags(intervals, t)
    Y, M = np.zeros((len(t), C), np.float32), np.zeros((len(t), C), np.float32)
    cache: dict[str, tuple | None] = {}
    bad = set()
    for k, tag in enumerate(tags):
        if not tag:
            continue
        if tag not in cache:
            cache[tag] = tag_to_target(tag, cfg)
            if cache[tag] is None:
                bad.add(tag)
        if cache[tag] is not None:
            Y[k], M[k] = cache[tag]
    for tag in sorted(bad):
        if tag not in ("free", "unlabeled"):
            warnings.warn(f"unknown label tag '{tag}' treated as unlabeled")
    return Y, M, tags


def frames_to_intervals(tags: np.ndarray, t: np.ndarray) -> list[tuple[float, float, str]]:
    """Run-length encode per-frame tag strings ('' = unlabeled) into intervals."""
    out = []
    if len(tags) == 0:
        return out
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    start = 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] != tags[start]:
            if tags[start]:
                out.append((float(t[start]), float(t[i - 1] + dt), str(tags[start])))
            start = i
    return out
