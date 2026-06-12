"""Environment factories with consistent seeding and optional normalisation.

make_vec_env_ wraps SB3's vectorised constructor; make_single is for evaluation
and rendering. normalize=True adds VecNormalize for the E2 observation-norm study.
"""
from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


def make_vec(env_id: str, n_envs: int = 4, seed: int = 0, normalize: bool = False):
    """Vectorised training env. Returns a VecEnv (optionally VecNormalize-wrapped)."""
    venv = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


def make_single(env_id: str, seed: int | None = None, render_mode: str | None = None):
    """Single env for evaluation, data collection, or rendering."""
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env
