"""Shared evaluation and video rendering.

evaluate() wraps SB3's evaluate_policy with the project's fixed protocol
(20 deterministic episodes). evaluate_torch() runs a bare nn.Module student.
record_side_by_side() produces the M6 expert-vs-student comparison video.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from . import config


def load_obs_normalizer(vecnorm_path):
    """Return a fn raw_obs -> normalised_obs from a saved VecNormalize pickle.

    When the expert was trained with VecNormalize, its policy consumes normalised
    observations, so BC students must be trained and evaluated on the same
    representation. Returns None if the path does not exist.
    """
    import pickle

    from pathlib import Path
    if not Path(vecnorm_path).exists():
        return None
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)
    vn.training = False
    return lambda o: vn.normalize_obs(np.asarray(o, dtype=np.float32)).astype(np.float32)


def evaluate(model, env_id: str, n_episodes: int = config.EVAL_EPISODES,
             seed: int = config.EVAL_SEED, vecnorm_path=None) -> tuple[float, float]:
    """Mean and std return for an SB3 model or policy.

    If vecnorm_path is given, the eval env is wrapped with the saved VecNormalize
    stats (obs normalised, reward raw) so a policy trained on normalised obs is
    evaluated correctly.
    """
    from pathlib import Path
    if vecnorm_path and Path(vecnorm_path).exists():
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        venv = DummyVecEnv([lambda: gym.make(env_id)])
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training, venv.norm_reward = False, False
        mean, std = evaluate_policy(model, venv, n_eval_episodes=n_episodes,
                                    deterministic=True)
        venv.close()
        return float(mean), float(std)
    env = gym.make(env_id)
    env.reset(seed=seed)
    mean, std = evaluate_policy(model, env, n_eval_episodes=n_episodes,
                                deterministic=True)
    env.close()
    return float(mean), float(std)


def evaluate_torch(student, env_id: str, device: str,
                   n_episodes: int = config.EVAL_EPISODES,
                   seed: int = config.EVAL_SEED, normalizer=None) -> tuple[float, float]:
    """Mean and std return for a bare torch nn.Module mapping obs -> action.

    If normalizer is given (from load_obs_normalizer), observations are normalised
    before the student, matching the representation the student was trained on.
    """
    import torch

    env = gym.make(env_id)
    student.eval()
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret = False, 0.0
        while not done:
            model_obs = obs if normalizer is None else normalizer(obs)
            obs_t = torch.as_tensor(model_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = student(obs_t).squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def record_video(predict_fn, env_id: str, out_path: Path, max_steps: int = 1000,
                 seed: int = 0) -> Path:
    """Render one episode to mp4. predict_fn maps obs -> action (clipped)."""
    from gymnasium.wrappers import RecordVideo

    folder = out_path.parent
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=str(folder), name_prefix=out_path.stem,
                      episode_trigger=lambda e: True)
    obs, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        action = predict_fn(obs)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()
    return out_path
