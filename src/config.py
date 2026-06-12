"""Project-wide constants and path resolution.

Paths resolve relative to PROJECT_ROOT, but every directory can be overridden
with an environment variable so the same notebook runs locally and on Colab
(point PROJECT_DATA_ROOT at a Google Drive mount there).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Reproducibility ---
SEEDS = [0, 1, 2, 3, 4]
EVAL_EPISODES = 20
EVAL_SEED = 99

# --- Environments ---
ENV_IDS = ["Walker2d-v4", "Ant-v4"]
RETURN_TARGETS = {"Walker2d-v4": 3000.0, "Ant-v4": 4000.0}

# --- Shared policy architecture (expert and student kept comparable) ---
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])  # SB3 v2.x dict form (no outer list)

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _dir(env_var: str, default: str) -> Path:
    root = Path(os.environ.get("PROJECT_DATA_ROOT", PROJECT_ROOT))
    path = Path(os.environ.get(env_var, root / default))
    path.mkdir(parents=True, exist_ok=True)
    return path


MODELS_DIR = _dir("PROJECT_MODELS_DIR", "models")
DATA_DIR = _dir("PROJECT_DATA_DIR", "data/demonstrations")
OUTPUTS_DIR = _dir("PROJECT_OUTPUTS_DIR", "outputs")
VIDEOS_DIR = _dir("PROJECT_VIDEOS_DIR", "videos")
LOGS_DIR = _dir("PROJECT_LOGS_DIR", "logs")


def device(prefer_gpu: bool = True) -> str:
    """Best available torch device. MPS helps from-scratch BC on Apple Silicon."""
    import torch

    if not prefer_gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
