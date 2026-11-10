from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "configs/default.yaml", overrides: list[str] | None = None) -> dict:
    """Load a YAML config. `overrides` are 'a.b.c=value' strings (value parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:  # inherit from another config, then deep-merge this one on top
        base = load_config(resolve_rel(path, cfg.pop("base")))
        cfg = _merge(base, cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_rel(cfg_path, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(cfg_path).resolve().parent / p


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def class_index(cfg: dict) -> dict[str, int]:
    return {name: i for i, name in enumerate(cfg["classes"])}


def clone(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
