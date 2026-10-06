"""Configuration loading.

`config.yaml` at the repo root is the single source of truth. Callers can pass
a different file and/or a dict of overrides (deep-merged on top).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(cfg, overrides or {})
    # YAML reads `{1: 50}` keys as ints already; ratings map stays str -> int.
    cfg["eqs"]["kinerja_points"] = {int(k): v for k, v in cfg["eqs"]["kinerja_points"].items()}
    return cfg
