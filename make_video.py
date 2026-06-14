"""M6: side-by-side expert vs BC-student video.

Renders one episode of the expert and one of the library BC student, stacks the
frames horizontally, and writes an mp4. Observations are normalised with the
expert's VecNormalize stats when present. Rendering is wrapped defensively;
MuJoCo offscreen rendering can fail headless.
Usage: make_video.py [ENV_ID] [DATA_KEY]
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from src import config, eval, seeding

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
DATA_KEY = sys.argv[2] if len(sys.argv) > 2 else ENV_ID
seeding.set_seed(0)
MODEL_DIR = config.MODELS_DIR / f"ppo_expert_{DATA_KEY}"
VN_PATH = MODEL_DIR / "vecnormalize.pkl"
norm = eval.load_obs_normalizer(VN_PATH)
N = (lambda x: norm(x)) if norm is not None else (lambda x: x)
print(f"[video] env={ENV_ID} data_key={DATA_KEY} normalized={norm is not None}", flush=True)


def rollout_frames(predict, seed, max_steps=1000):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps):
        frames.append(env.render())
        action = predict(obs)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()
    return frames


def expert_predict(obs):
    a, _ = expert.predict(N(obs), deterministic=True)
    return a


def student_predict(obs):
    a, _ = student.predict(N(obs), deterministic=True)
    return a


try:
    expert = PPO.load(MODEL_DIR / "best_model", device="cpu")
    student = ActorCriticPolicy.load(
        str(config.MODELS_DIR / "bc_student" / f"bc_student_imitation_{DATA_KEY}"),
        device="cpu")
    fe = rollout_frames(expert_predict, seed=0)
    fs = rollout_frames(student_predict, seed=0)
    n = min(len(fe), len(fs))
    fe, fs = fe[:n], fs[:n]
    sep = np.ones((fe[0].shape[0], 8, 3), dtype=np.uint8) * 255
    combined = [np.concatenate([e, sep, s], axis=1) for e, s in zip(fe, fs)]

    from moviepy.editor import ImageSequenceClip
    out = config.VIDEOS_DIR / f"expert_vs_student_{DATA_KEY}.mp4"
    ImageSequenceClip(combined, fps=30).write_videofile(str(out), logger=None)
    print(f"[video] DONE -> {out} ({n} frames, expert|student)", flush=True)
except Exception as e:
    print(f"[video] FAILED for {DATA_KEY}: {type(e).__name__}: {e}", flush=True)
