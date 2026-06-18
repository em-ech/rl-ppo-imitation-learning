"""M6: side-by-side expert vs BC-student video.

Renders one deterministic episode of the PPO expert and one of the library BC
student, draws a title bar above each panel, stacks them horizontally, and writes
an mp4. Observations are normalised with the expert's VecNormalize stats when
present. Rendering is wrapped defensively; MuJoCo offscreen rendering can fail
headless.
Usage: make_video.py [ENV_ID] [DATA_KEY]
"""
import sys

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from src import config, eval, seeding, video

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
DATA_KEY = sys.argv[2] if len(sys.argv) > 2 else ENV_ID
seeding.set_seed(0)
MODEL_DIR = config.MODELS_DIR / f"ppo_expert_{DATA_KEY}"
VN_PATH = MODEL_DIR / "vecnormalize.pkl"
norm = eval.load_obs_normalizer(VN_PATH)
N = (lambda x: norm(x)) if norm is not None else (lambda x: x)
print(f"[video] env={ENV_ID} data_key={DATA_KEY} normalized={norm is not None}", flush=True)


def rollout(model, seed=0, max_steps=1000):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, ret = [], 0.0
    for _ in range(max_steps):
        frames.append(env.render())
        action, _ = model.predict(N(obs), deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        ret += r
        if term or trunc:
            break
    env.close()
    return frames, ret


try:
    expert = PPO.load(MODEL_DIR / "best_model", device="cpu")
    student = ActorCriticPolicy.load(
        str(config.MODELS_DIR / "bc_student" / f"bc_student_imitation_{DATA_KEY}"),
        device="cpu")
    fe, re = rollout(expert)
    fs, rs = rollout(student)
    tracks = [video.titled_track(fe, f"Expert (PPO)  ({re:.0f})"),
              video.titled_track(fs, f"BC student  ({rs:.0f})")]
    combined = video.stack_panels(tracks)

    out = config.VIDEOS_DIR / f"expert_vs_student_{DATA_KEY}.mp4"
    video.write_mp4(combined, out)
    print(f"[video] DONE -> {out} ({len(combined)} frames; expert {re:.0f} | student {rs:.0f})",
          flush=True)
except Exception as e:
    print(f"[video] FAILED for {DATA_KEY}: {type(e).__name__}: {e}", flush=True)
