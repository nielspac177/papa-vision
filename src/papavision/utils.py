"""Shared utilities: device selection, reproducible seeding, config loading, logging.

These helpers are deliberately dependency-light so they can be imported by every
other module without creating import cycles.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = two levels up from this file (src/papavision/utils.py -> repo).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


def ensure_dirs(*paths: Path) -> None:
    """Create each directory (and parents) if it does not already exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "papavision") -> logging.Logger:
    """Return a process-wide logger with a single, clean stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that can influence training for reproducible runs.

    We seed Python ``random``, NumPy, and PyTorch (CPU + all accelerators) and,
    when ``deterministic`` is set, request deterministic algorithms. Apple MPS
    does not expose a fully deterministic backend, so on MPS we still fix all
    seeds (which removes the dominant source of run-to-run variance) but do not
    force ``torch.use_deterministic_algorithms`` to avoid hard errors.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Explicitly seed the MPS generator too (the primary device on Apple Silicon),
    # mirroring the CUDA path rather than relying on global-generator propagation.
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if deterministic:
        # cuDNN knobs are harmless no-ops on CPU/MPS.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    """Select the best available device.

    Order of preference for ``"auto"``: Apple MPS -> CUDA -> CPU. A specific
    string (``"cpu"``, ``"mps"``, ``"cuda"``) forces that device when available
    and otherwise falls back to CPU with a warning.
    """
    log = get_logger()
    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if prefer == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if prefer == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(prefer)


# --------------------------------------------------------------------------- #
# Config + JSON I/O
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return cfg


def _json_default(obj: Any) -> Any:
    """Make NumPy / Path / dataclass objects JSON-serializable."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` to ``path`` as pretty-printed JSON (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    """Load JSON from ``path``."""
    with open(path, "r") as f:
        return json.load(f)
