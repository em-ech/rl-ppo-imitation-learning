"""Expert demonstration collection and dataset loading.

Fixes a gap in the provided listings: per-episode boundaries are preserved via
an `episode_starts` index and `episode_lengths` array, which the dataset-size
ablation (M5) and the trajectory-length histogram both require. Actions are the
deterministic policy mean, as the spec instructs.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np


def collect(model, env_id: str, n_episodes: int, out_dir: Path,
            seed: int = 0, obs_transform=None) -> dict:
    """Roll out the expert for n_episodes and save the dataset to out_dir.

    Saves observations.npy, actions.npy (flat, concatenated across episodes),
    episode_returns.npy, episode_lengths.npy, and episode_starts.npy (the index
    of the first transition of each episode in the flat arrays).

    obs_transform, if given, is applied to each observation *before* the expert
    predicts (e.g. VecNormalize obs normalisation for the Ant expert). The RAW
    observation is still what gets stored, so the BC student learns a
    raw-observation -> action mapping and needs no normalisation at deployment.
    """
    env = gym.make(env_id)
    all_obs, all_acts, returns, lengths = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret = False, 0.0
        ep_obs, ep_acts = [], []
        while not done:
            model_obs = obs if obs_transform is None else obs_transform(obs)
            action, _ = model.predict(model_obs, deterministic=True)
            ep_obs.append(obs.copy())
            ep_acts.append(action.copy())
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            done = term or trunc
        all_obs.append(np.asarray(ep_obs, dtype=np.float32))
        all_acts.append(np.asarray(ep_acts, dtype=np.float32))
        returns.append(ep_ret)
        lengths.append(len(ep_obs))
    env.close()

    lengths = np.asarray(lengths, dtype=np.int64)
    episode_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    obs_arr = np.concatenate(all_obs, axis=0)
    acts_arr = np.concatenate(all_acts, axis=0)
    returns = np.asarray(returns, dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "observations.npy", obs_arr)
    np.save(out_dir / "actions.npy", acts_arr)
    np.save(out_dir / "episode_returns.npy", returns)
    np.save(out_dir / "episode_lengths.npy", lengths)
    np.save(out_dir / "episode_starts.npy", episode_starts)

    return {
        "observations": obs_arr,
        "actions": acts_arr,
        "episode_returns": returns,
        "episode_lengths": lengths,
        "episode_starts": episode_starts,
    }


def load(data_dir: Path) -> dict:
    """Load a saved demonstration dataset."""
    return {name: np.load(data_dir / f"{name}.npy") for name in
            ["observations", "actions", "episode_returns",
             "episode_lengths", "episode_starts"]}


def subset(data: dict, n_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (obs, acts) for the first n_episodes, using episode_starts.

    Used by the dataset-size ablation so subsets respect episode boundaries.
    """
    starts = data["episode_starts"]
    lengths = data["episode_lengths"]
    n_episodes = min(n_episodes, len(lengths))
    end = starts[n_episodes - 1] + lengths[n_episodes - 1]
    return data["observations"][:end], data["actions"][:end]
