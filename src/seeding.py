"""Single seeding entry point for numpy, torch, and Python random.

The spec penalises non-reproducible runs, so every experiment calls set_seed
once at the top. SB3 takes its own `seed=` argument, and gymnasium envs take a
seed in reset(); this covers the rest.
"""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
