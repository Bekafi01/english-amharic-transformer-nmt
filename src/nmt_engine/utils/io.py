"""File and configuration I/O helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Ensures a directory exists, creating all parent directories if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads and parses a YAML file safely."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found at: {p.resolve()}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Saves a dictionary to a YAML file, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    """Loads and parses a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found at: {p.resolve()}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Saves data to a JSON file with utf-8 encoding and indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    return p
