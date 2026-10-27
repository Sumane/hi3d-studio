"""Paths and settings written by install.sh."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
OUTPUTS = ROOT / "outputs"
ENGINES = ROOT / "engines"


def load() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    cfg.setdefault("port", 7860)
    cfg.setdefault("gpu_index", 0)
    cfg.setdefault("hi3d_model", str(ROOT / "models" / "Hi3Dv1.0"))
    cfg.setdefault("hf_home", str(ROOT / "hf-cache"))
    return cfg


def gpu_env(cfg: dict) -> dict:
    """Pin child processes to the configured card, numbered as nvidia-smi shows it."""
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(cfg["gpu_index"]),
        "HF_HOME": cfg["hf_home"],
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Off the (nearly full) system drive.
        "TRITON_CACHE_DIR": str(ROOT / ".cache" / "triton"),
        "U2NET_HOME": str(ROOT / ".cache" / "u2net"),
        "TMPDIR": str(ROOT / ".cache" / "tmp"),
    }
