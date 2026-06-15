"""Stage 5: imitation as PPO pretraining (sample efficiency).

Compares eval-return-vs-environment-timesteps for: PPO from scratch, BC+PPO
fine-tune, DAgger+PPO fine-tune, with BC-only and DAgger-only as references.
Warm-started runs load the expert's VecNormalize stats so the imitation policy
sees a consistent observation distribution. CPU-only.
Usage: pretraining.py [ENV_ID] [DATA_KEY] [PROFILE] [TIMESTEPS] [N_ENVS]
"""
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize

from src import config, envs, plotting, seeding
from src.ppo import PROFILES

torch.set_num_threads(1)
ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
DATA_KEY = sys.argv[2] if len(sys.argv) > 2 else ENV_ID
PROFILE = sys.argv[3] if len(sys.argv) > 3 else "default"
TIMESTEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
N_ENVS = int(sys.argv[5]) if len(sys.argv) > 5 else (1 if PROFILE == "tuned_ant" else 4)
seeding.set_seed(0)
t0 = time.time()

MODEL_DIR = config.MODELS_DIR / f"ppo_expert_{DATA_KEY}"
VN_PATH = MODEL_DIR / "vecnormalize.pkl"
NORM = VN_PATH.exists()
print(f"[pretrain] env={ENV_ID} data_key={DATA_KEY} profile={PROFILE} "
      f"steps={TIMESTEPS:,} n_envs={N_ENVS} normalized={NORM}", flush=True)


def make_train_env(load_expert_stats):
    venv = envs.make_vec(ENV_ID, n_envs=N_ENVS, seed=0)
    if NORM:
        if load_expert_stats:
            venv = VecNormalize.load(str(VN_PATH), venv)
            venv.training, venv.norm_reward = True, True
        else:
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def make_eval_env():
    ev = envs.make_vec(ENV_ID, n_envs=1, seed=99)
    if NORM:
        ev = VecNormalize(ev, norm_obs=True, norm_reward=False, clip_obs=10.0,
                          training=False)
    return ev


def train_curve(name, init_policy_zip=None):
    """Train PPO (optionally warm-started) and return (timesteps, returns)."""
    venv = make_train_env(load_expert_stats=init_policy_zip is not None)
    model = PPO("MlpPolicy", venv, verbose=0, seed=0, device="cpu",
                policy_kwargs=dict(net_arch=config.NET_ARCH), **PROFILES[PROFILE])
    if init_policy_zip is not None:
        p = ActorCriticPolicy.load(str(init_policy_zip), device="cpu")
        model.policy.load_state_dict(p.state_dict(), strict=False)
    log_dir = config.LOGS_DIR / f"pretrain_{DATA_KEY}_{name}"
    cb = EvalCallback(make_eval_env(), log_path=str(log_dir),
                      eval_freq=max(10_000 // N_ENVS, 1), n_eval_episodes=5,
                      deterministic=True, render=False)
    model.learn(total_timesteps=TIMESTEPS, callback=cb, progress_bar=False)
    d = np.load(log_dir / "evaluations.npz")
    ts, ret = d["timesteps"].tolist(), d["results"].mean(axis=1).tolist()
    print(f"[pretrain] {name}: final {ret[-1]:.1f} (best {max(ret):.1f})  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ts, ret


# SB3 policy.save() writes without a .zip extension.
bc_zip = config.MODELS_DIR / "bc_student" / f"bc_student_imitation_{DATA_KEY}"
dagger_zip = config.MODELS_DIR / "dagger_student" / f"dagger_student_{DATA_KEY}"
results = {"env": ENV_ID, "data_key": DATA_KEY, "profile": PROFILE,
           "timesteps": TIMESTEPS, "curves": {}}

# Curves (each isolated: best judgment, skip on failure).
for name, init in [("scratch", None), ("bc_ppo", bc_zip), ("dagger_ppo", dagger_zip)]:
    if init is not None and not Path(init).exists():
        print(f"[pretrain] SKIP {name}: missing {init}", flush=True)
        continue
    try:
        ts, ret = train_curve(name, init)
        results["curves"][name] = {"timesteps": ts, "returns": ret}
    except Exception as e:
        print(f"[pretrain] FAILED {name}: {e}", flush=True)

# Reference returns (BC-only, DAgger-only, expert).
def _final(json_path, key):
    p = config.OUTPUTS_DIR / json_path
    if p.exists():
        return json.load(open(p)).get(key)
    return None


bc_res = config.OUTPUTS_DIR / f"bc_results_{DATA_KEY}.json"
expert_mean = json.load(open(bc_res)).get("expert_mean") if bc_res.exists() else None
bc_only = json.load(open(bc_res)).get("library_bc", {}).get("mean") if bc_res.exists() else None
dagger_res = config.OUTPUTS_DIR / f"dagger_results_{DATA_KEY}.json"
dagger_only = (json.load(open(dagger_res)).get("returns_by_iter", [None])[-1]
               if dagger_res.exists() else None)
results.update(expert_mean=expert_mean, bc_only=bc_only, dagger_only=dagger_only)

# Plot: return vs env timesteps.
fig, ax = plt.subplots(figsize=(9, 6))
colors = {"scratch": "gray", "bc_ppo": "steelblue", "dagger_ppo": "darkorange"}
labels = {"scratch": "PPO from scratch", "bc_ppo": "BC + PPO", "dagger_ppo": "DAgger + PPO"}
for name, c in results["curves"].items():
    ax.plot(c["timesteps"], c["returns"], color=colors.get(name), label=labels.get(name, name))
if bc_only:
    ax.axhline(bc_only, color="steelblue", linestyle=":", alpha=0.7, label=f"BC only ({bc_only:.0f})")
if dagger_only:
    ax.axhline(dagger_only, color="darkorange", linestyle=":", alpha=0.7, label=f"DAgger only ({dagger_only:.0f})")
if expert_mean:
    ax.axhline(expert_mean, color="green", linestyle="--", label=f"Expert ({expert_mean:.0f})")
ax.set_xlabel("Environment timesteps"); ax.set_ylabel("Mean Evaluation Return")
ax.set_title(f"Imitation as PPO Pretraining ({DATA_KEY})"); ax.legend(); ax.grid(True, alpha=0.3)
plotting.save(fig, config.OUTPUTS_DIR / f"pretraining_{DATA_KEY}.png")
with open(config.OUTPUTS_DIR / f"pretraining_results_{DATA_KEY}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[pretrain] DONE in {(time.time()-t0)/60:.1f} min -> "
      f"outputs/pretraining_results_{DATA_KEY}.json", flush=True)
