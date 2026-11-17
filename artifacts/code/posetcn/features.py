"""Landmarks -> per-frame posture features.

Pipeline
--------
1. `canonical_points`: pick the camera-facing body side (left or right landmarks)
   by visibility, convert to pixel space with y pointing UP, and mirror so the
   person always faces +x. Results: 7 keypoints (T, 7, 2), visibility, detected mask.
2. `points_to_features`: hip-centred, torso-normalised coordinates + sagittal
   angles + deltas. This function is the one the mobile app must port exactly.

Angle conventions (degrees, after canonicalisation):
  neck_angle   ear relative to shoulder, measured from vertical, + = head forward
  trunk_angle  shoulder relative to hip, measured from vertical,  + = leaning forward
  hip_angle    interior angle shoulder-hip-knee (about 90-110 when sitting upright)
  head_tilt    nose relative to ear vs horizontal, + = chin up
  thigh_angle  hip->knee below horizontal, + = knee lower than hip (perched on the seat edge)
  knee_angle   interior angle hip-knee-ankle (small = feet tucked under the seat)
  shoulder_line  signed distance of the shoulder from the ear-hip line / torso, + = shoulder ahead
                 (rounded shoulders)
"""
from __future__ import annotations

import numpy as np

# MediaPipe Pose (33 landmarks) indices
KP_NAMES = ["nose", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
LEFT_IDX = [0, 7, 11, 13, 15, 23, 25, 27]
RIGHT_IDX = [0, 8, 12, 14, 16, 24, 26, 28]
NOSE, EAR, SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE = range(8)

GEOM_NAMES = [
    "neck_angle", "trunk_angle", "hip_angle", "head_tilt",
    "head_fwd", "shoulder_fwd", "ear_height", "torso_ratio",
    "thigh_angle", "knee_angle", "shoulder_line",
]
REL_KPS = [NOSE, EAR, SHOULDER, ELBOW, WRIST, KNEE, ANKLE]
VIS_KPS = [EAR, SHOULDER, HIP, KNEE, ANKLE]
FEATURE_NAMES = (
    GEOM_NAMES
    + [f"{KP_NAMES[k]}_{a}" for k in REL_KPS for a in ("x", "y")]
    + [f"vis_{KP_NAMES[k]}" for k in VIS_KPS]
    + ["detected"]
    + [f"d_{n}" for n in GEOM_NAMES]
)
NUM_FEATURES = len(FEATURE_NAMES)


def choose_side(lm: np.ndarray) -> str:
    """'left' or 'right': which side faces the camera (higher mean visibility)."""
    left = np.nanmean(lm[:, [7, 11, 23], 3]) if np.isfinite(lm[:, [7, 11, 23], 3]).any() else 0.0
    right = np.nanmean(lm[:, [8, 12, 24], 3]) if np.isfinite(lm[:, [8, 12, 24], 3]).any() else 0.0
    return "left" if left >= right else "right"


def _ffill_bfill(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid frames (axis 0) with the last valid value, then the next valid one."""
    if valid.all():
        return a
    if not valid.any():
        return np.zeros_like(a)
    idx = np.where(valid, np.arange(len(a)), -1)
    idx = np.maximum.accumulate(idx)
    first = np.argmax(valid)
    idx[idx < 0] = first
    return a[idx]


def canonical_points(
    lm: np.ndarray,
    width: int,
    height: int,
    side: str | None = None,
    facing: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """lm: (T, 33, 4) normalised [x, y, z, visibility] with NaN rows for undetected frames.

    Returns pts (T,8,2) in pixels (y up, facing +x), vis (T,8), detected (T,), meta.
    Pass `side`/`facing` to fix them (e.g. for live streaming once estimated).
    """
    lm = np.asarray(lm, dtype=np.float64)
    detected = np.isfinite(lm[:, 0, 0])
    side = side or choose_side(lm)
    idx = LEFT_IDX if side == "left" else RIGHT_IDX

    pts = np.stack([lm[:, idx, 0] * width, -lm[:, idx, 1] * height], axis=-1)
    vis = np.nan_to_num(lm[:, idx, 3], nan=0.0)

    if facing is None:
        dx = pts[detected, NOSE, 0] - pts[detected, EAR, 0]
        facing = 1.0 if (len(dx) == 0 or np.median(dx) >= 0) else -1.0
    pts[..., 0] *= facing

    pts = _ffill_bfill(pts, detected)
    vis = np.where(detected[:, None], vis, 0.0)
    return pts.astype(np.float32), vis.astype(np.float32), detected.astype(np.float32), {
        "side": side, "facing": float(facing)}


def torso_length(pts: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pts[..., SHOULDER, :] - pts[..., HIP, :], axis=-1)


def reference_scale(pts: np.ndarray, detected: np.ndarray) -> float:
    """Typical torso length for a clip (median over detected frames)."""
    t = torso_length(pts)
    t = t[detected > 0.5] if (detected > 0.5).any() else t
    return float(np.median(t)) if len(t) else 1.0


def _angle_from_vertical(v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(v[..., 0], v[..., 1]))


def _interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def geometry(pts: np.ndarray, ref_scale: float | None = None) -> np.ndarray:
    """Interpretable per-frame geometry (T, 11) in natural units (degrees, torso lengths)."""
    E, S, H, K, N, A = pts[:, EAR], pts[:, SHOULDER], pts[:, HIP], pts[:, KNEE], pts[:, NOSE], pts[:, ANKLE]
    torso = np.maximum(torso_length(pts), 1e-6)
    ref = ref_scale if ref_scale else float(np.median(torso))
    eh = E - H
    eh_len = np.linalg.norm(eh, axis=-1) + 1e-6
    # 2D cross product: + when the shoulder lies ahead (+x side) of the hip->ear line
    shoulder_line = (eh[:, 1] * (S - H)[:, 0] - eh[:, 0] * (S - H)[:, 1]) / eh_len / torso
    return np.stack([
        _angle_from_vertical(E - S),
        _angle_from_vertical(S - H),
        _interior_angle(S, H, K),
        np.degrees(np.arctan2(N[:, 1] - E[:, 1], np.abs(N[:, 0] - E[:, 0]) + 1e-6)),
        (E[:, 0] - S[:, 0]) / torso,
        (S[:, 0] - H[:, 0]) / torso,
        (E[:, 1] - H[:, 1]) / torso,
        torso / max(ref, 1e-6),
        np.degrees(np.arctan2(H[:, 1] - K[:, 1], np.abs(K[:, 0] - H[:, 0]) + 1e-6)),
        _interior_angle(H, K, A),
        shoulder_line,
    ], axis=-1)


def points_to_features(
    pts: np.ndarray, vis: np.ndarray, detected: np.ndarray, ref_scale: float | None = None
) -> np.ndarray:
    """(T,8,2),(T,8),(T,) -> (T, NUM_FEATURES) float32. Deltas are causal (x_t - x_{t-1})."""
    g = geometry(pts, ref_scale)
    # rescale to roughly unit range
    g_scaled = np.stack([
        g[:, 0] / 90.0, g[:, 1] / 90.0, (g[:, 2] - 90.0) / 90.0, g[:, 3] / 90.0,
        g[:, 4], g[:, 5], g[:, 6], g[:, 7] - 1.0,
        g[:, 8] / 90.0, (g[:, 9] - 90.0) / 90.0, g[:, 10],
    ], axis=-1)

    torso = np.maximum(torso_length(pts), 1e-6)[:, None]
    rel = (pts[:, REL_KPS] - pts[:, HIP:HIP + 1]) / torso[..., None]
    rel = rel.reshape(len(pts), -1)

    delta = np.zeros_like(g_scaled)
    delta[1:] = g_scaled[1:] - g_scaled[:-1]

    feats = np.concatenate([g_scaled, rel, vis[:, VIS_KPS], detected[:, None], delta], axis=-1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def landmarks_to_features(lm: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
    """Convenience: full clip landmarks -> features."""
    pts, vis, det, meta = canonical_points(lm, width, height)
    ref = reference_scale(pts, det)
    meta["ref_scale"] = ref
    return points_to_features(pts, vis, det, ref), meta
