"""Turns per-frame fault probabilities into haptic alerts, one buzzer per fault.

Port this logic to the mobile app as-is. For each fault independently: it fires when
prob >= threshold continuously for `persist_s`, then that fault waits `cooldown_s`.
While the transient (gate) output is on, all timers reset (no buzzing mid-drink).
Several faults may fire on the same frame (e.g. neck + shoulders).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Alert:
    t: float
    cls: str
    zone: str


class AlertEngine:
    def __init__(self, cfg: dict):
        a = cfg["alerts"]
        self.classes = cfg["classes"]
        self.zones = cfg["alert_zones"]
        self.fault_idx = [self.classes.index(c) for c in self.zones]
        gate = cfg.get("gate_class")
        self.gate_idx = self.classes.index(gate) if gate else None
        per = a.get("per_class_threshold") or {}
        self.thr = {i: per.get(self.classes[i], a["prob_threshold"]) for i in self.fault_idx}
        self.gate_thr, self.persist, self.cooldown = a["gate_threshold"], a["persist_s"], a["cooldown_s"]
        self.reset()

    def reset(self):
        self.since: dict[int, float | None] = {i: None for i in self.fault_idx}
        self.last: dict[int, float] = {i: -1e9 for i in self.fault_idx}

    def update(self, t: float, probs: np.ndarray) -> list[Alert]:
        if self.gate_idx is not None and probs[self.gate_idx] >= self.gate_thr:
            self.since = {i: None for i in self.fault_idx}
            return []
        fired = []
        for i in self.fault_idx:
            if probs[i] < self.thr[i]:
                self.since[i] = None
                continue
            if self.since[i] is None:
                self.since[i] = t
            if t - self.since[i] >= self.persist and t - self.last[i] >= self.cooldown:
                c = self.classes[i]
                fired.append(Alert(t, c, self.zones[c]))
                self.last[i] = t
        return fired


def simulate(cfg: dict, t: np.ndarray, probs: np.ndarray, Y: np.ndarray, M: np.ndarray) -> dict:
    """Replay a clip through the alert engine and score it against frame targets.

    false alert = alert for fault c at a frame where c is KNOWN to be absent
    latency     = time from the start of a labeled fault-c run to its first c alert
    missed      = labeled fault-c run longer than persist_s + 2 s with no c alert
    """
    eng = AlertEngine(cfg)
    alerts = []
    for k in range(len(t)):
        alerts += [(k, a) for a in eng.update(float(t[k]), probs[k])]
    ci = {c: i for i, c in enumerate(cfg["classes"])}
    false = sum(1 for k, a in alerts if M[k, ci[a.cls]] > 0 and Y[k, ci[a.cls]] < 0.5)

    latencies, missed = [], 0
    for c in cfg["alert_zones"]:
        i = ci[c]
        on = (Y[:, i] > 0.5) & (M[:, i] > 0)
        k = 0
        while k < len(on):
            if not on[k]:
                k += 1
                continue
            j = k
            while j < len(on) and on[j]:
                j += 1
            hits = [t[kk] for kk, a in alerts if a.cls == c and k <= kk < j]
            if hits:
                latencies.append(float(hits[0] - t[k]))
            elif t[j - 1] - t[k] > cfg["alerts"]["persist_s"] + 2:
                missed += 1
            k = j
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    labeled_minutes = float((M.max(1) > 0).sum()) * dt / 60
    return {"alerts": len(alerts), "false_alerts": false, "labeled_minutes": labeled_minutes,
            "latencies": latencies, "missed_segments": missed}
