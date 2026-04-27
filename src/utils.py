"""
utils.py — Shared utilities for MeloMatch.

Centralizes config loading, path management, seed setting, and logging.
"""

import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import yaml


# ======================== Config ========================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


# ======================== Reproducibility ========================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ======================== I/O ========================

def load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str):
    """Save records to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ======================== Logging ========================

def setup_logging(
    log_dir: str = "results",
    run_name: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up structured logging to both console and file.
    Returns the root logger.
    """
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    log_path = Path(log_dir) / run_name / "experiment.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("melomatch")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to {log_path}")
    return logger


# ======================== Experiment Tracking ========================

def save_run_metadata(run_dir: str, config: dict, extra: Optional[dict] = None):
    """Save experiment metadata for reproducibility (à la ONCE's config pattern).

    Redacts API keys to prevent accidental exposure in shared results.
    """
    import copy
    safe_config = copy.deepcopy(config)
    # Redact any api_key fields at any nesting depth
    _redact_keys(safe_config)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "config": safe_config,
    }
    if extra:
        meta.update(extra)

    out = Path(run_dir) / "run_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _redact_keys(d: dict):
    """Recursively redact fields named 'api_key' or 'secret' in a dict."""
    for key in list(d.keys()):
        if isinstance(d[key], dict):
            _redact_keys(d[key])
        elif key in ("api_key", "secret", "password", "token") and d[key]:
            d[key] = "<REDACTED>"
