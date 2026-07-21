"""
utils/config.py — load and expose config/config.yaml as a simple namespace.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Load YAML config. Falls back to config/config.yaml relative to project root."""
    if path is None:
        # Resolve relative to *this file* so it works regardless of cwd
        path = Path(__file__).parent.parent / "config" / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return _to_namespace(raw)


# Module-level singleton — importers may use `from utils.config import CFG`
CFG = load_config()