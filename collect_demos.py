"""Collect expert demonstrations + EDA + quality gate (M2). Mirrors notebook 02.

If a vecnormalize.pkl sits beside the expert (the normalised Ant case), its obs
normalisation is applied before the expert predicts, while raw observations are
stored. Usage: .venv/bin/python collect_demos.py [ENV_ID] [N_EPISODES]
"""
import pickle
import sys

import numpy as np
from stable_baselines3 import PPO

from src import collect, config, plotting, seeding

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
N_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 100
seeding.set_seed(0)

MODEL_DIR = config.MODELS_DIR / f"ppo_expert_{ENV_ID}"
model = PPO.load(MODEL_DIR / "best_model")

obs_transform = None
vecnorm = MODEL_DIR / "vecnormalize.pkl"
if vecnorm.exists():
    with open(vecnorm, "rb") as f:
        vn = pickle.load(f)
    vn.training = False
    obs_transform = lambda o: vn.normalize_obs(o).astype(np.float32)
    print(f"[collect] applying VecNormalize obs stats from {vecnorm.name}", flush=True)

out_dir = config.DATA_DIR / ENV_ID
data = collect.collect(model, ENV_ID, n_episodes=N_EPISODES, out_dir=out_dir,
                       seed=0, obs_transform=obs_transform)

r = data["episode_returns"]
thr = (2 / 3) * r.mean()
frac = float(np.mean(r > thr))
print(f"[collect] {ENV_ID}: {len(r)} eps, {len(data['observations']):,} transitions",
      flush=True)
print(f"[collect] mean return {r.mean():.1f} +/- {r.std():.1f} | "
      f"min/max {r.min():.1f}/{r.max():.1f} | mean len {data['episode_lengths'].mean():.1f}",
      flush=True)
print(f"[collect] quality gate: {100*frac:.1f}% above 2/3-mean ({thr:.0f}) -> "
      f"{'PASS' if frac >= 0.9 else 'FAIL'}", flush=True)

plotting.save(plotting.dataset_eda(r, data["actions"]),
              config.OUTPUTS_DIR / f"dataset_analysis_{ENV_ID}.png")
print(f"[collect] saved EDA -> outputs/dataset_analysis_{ENV_ID}.png", flush=True)
