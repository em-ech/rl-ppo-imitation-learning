"""Shared utilities for the PPO + Imitation Learning group project.

Submodules are imported lazily so lightweight helpers such as config, seeding,
and collect can be used without immediately importing the full SB3/MuJoCo stack.
"""
from importlib import import_module

__all__ = ["bc_bridge", "bc_scratch", "collect", "config", "dagger", "envs",
           "eval", "plotting", "ppo", "sac", "seeding", "video"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
