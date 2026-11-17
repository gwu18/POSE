"""Multi-label metrics (no sklearn).

Every metric for output c only uses windows where that output is known (mask = 1).
"good posture" is derived: a window is truly good if all fault outputs are known 0, and
predicted good if every fault probability is below its threshold.
"""
from __future__ import annotations

import numpy as np


def auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = p[y > 0.5], p[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def thresholds_for(cfg: dict) -> np.ndarray:
    a = cfg.get("alerts", {})
    per = a.get("per_class_threshold") or {}
    base = 0.5  # metric threshold; alert threshold is separate (stricter)
    return np.array([per.get(c, base) for c in cfg["classes"]])


def report(Y: np.ndarray, P: np.ndarray, M: np.ndarray, cfg: dict, thr: np.ndarray | None = None) -> dict:
    """Y,M (N,C) targets/mask, P (N,C) probabilities."""
    classes = cfg["classes"]
    gate = cfg.get("gate_class")
    thr = thresholds_for(cfg) if thr is None else thr
    pred = (P >= thr).astype(np.float32)
    per = {}
    for i, c in enumerate(classes):
        k = M[:, i] > 0
        y, yh = Y[k, i], pred[k, i]
        tp, fp, fn = float((y * yh).sum()), float(((1 - y) * yh).sum()), float((y * (1 - yh)).sum())
        p, r, f = _prf(tp, fp, fn)
        per[c] = {"precision": p, "recall": r, "f1": f, "auroc": auroc(y, P[k, i]),
                  "positives": int(y.sum()), "known": int(k.sum())}

    fault_idx = [i for i, c in enumerate(classes) if c != gate]
    fault_f1 = [per[classes[i]]["f1"] for i in fault_idx if per[classes[i]]["positives"] > 0]

    # derived "good posture" (all faults known and 0)
    allk = M[:, fault_idx].min(1) > 0
    true_good = allk & (Y[:, fault_idx].max(1) < 0.5)
    pred_good = pred[:, fault_idx].max(1) < 0.5
    if gate is not None:
        gi = classes.index(gate)
        true_good &= ~((M[:, gi] > 0) & (Y[:, gi] > 0.5))
    tp = float((true_good & pred_good & allk).sum())
    fp = float((~true_good & pred_good & allk).sum())
    fn = float((true_good & ~pred_good & allk).sum())
    gp, gr, gf = _prf(tp, fp, fn)
    exact = float((pred[allk][:, fault_idx] == Y[allk][:, fault_idx]).all(1).mean()) if allk.any() else 0.0

    return {
        "mean_fault_f1": float(np.mean(fault_f1)) if fault_f1 else 0.0,
        "good": {"precision": gp, "recall": gr, "f1": gf, "support": int(true_good.sum())},
        "exact_match": exact,
        "per_class": per,
    }


def format_report(r: dict, classes: list[str]) -> str:
    lines = [f"mean fault F1 {r['mean_fault_f1']:.3f}   exact-match {r['exact_match']:.3f}", "",
             f"{'output':<19}{'prec':>7}{'rec':>7}{'f1':>7}{'auroc':>7}{'pos':>8}{'known':>8}"]
    for c in classes:
        p = r["per_class"][c]
        au = f"{p['auroc']:.3f}" if p["auroc"] is not None else "  -  "
        lines.append(f"{c:<19}{p['precision']:>7.3f}{p['recall']:>7.3f}{p['f1']:>7.3f}{au:>7}"
                     f"{p['positives']:>8}{p['known']:>8}")
    g = r["good"]
    lines.append(f"{'(good posture)':<19}{g['precision']:>7.3f}{g['recall']:>7.3f}{g['f1']:>7.3f}{'':>7}"
                 f"{g['support']:>8}")
    return "\n".join(lines)
