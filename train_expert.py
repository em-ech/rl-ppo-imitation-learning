"""Full PPO expert training (spec Section 3, M1).

Mirrors notebooks/01_ppo_expert.ipynb so a long run can go in the background.
Usage:
  python train_expert.py [ENV_ID] [TOTAL_TIMESTEPS] [N_ENVS] [norm] [profile] [resume]

  norm     enable VecNormalize obs normalisation (saved next to best_model).
  profile  'default' (our validated config) or 'tuned_ant' (rl-zoo3 Optuna
           hyperparameters for Ant; designed for n_envs=1, ~1e7 steps).
  resume   continue from the latest ppo_checkpoint in the model dir, so an
           interrupted multi-hour run does not restart from zero.
"""
import re
import sys
import time

import torch
# SB3 PPO with a small MLP on CPU is far faster single-threaded: with many
# threads, BLAS oversubscription on the tiny net thrashes (observed ~9 cores at
# <40 steps/s for single-env Ant). One thread avoids the contention.
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from src import config, envs, eval, seeding

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
TOTAL_TIMESTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
N_ENVS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
NORMALIZE = len(sys.argv) > 4 and sys.argv[4].lower() in ("1", "true", "norm", "normalize")
PROFILE = sys.argv[5] if len(sys.argv) > 5 else "default"
RESUME = len(sys.argv) > 6 and sys.argv[6].lower() in ("1", "true", "resume")

seeding.set_seed(0)
MODEL_DIR = config.MODELS_DIR / f"ppo_expert_{ENV_ID}"
LOG_DIR = config.LOGS_DIR / f"ppo_expert_{ENV_ID}"
VECNORM_PATH = MODEL_DIR / "vecnormalize.pkl"
t0 = time.time()
print(f"[expert] env={ENV_ID} steps={TOTAL_TIMESTEPS:,} n_envs={N_ENVS} "
      f"normalize={NORMALIZE} profile={PROFILE} resume={RESUME} -> {MODEL_DIR}",
      flush=True)


def linear_schedule(initial: float):
    """LR decays linearly from `initial` to 0 over training (progress 1 -> 0)."""
    return lambda progress_remaining: progress_remaining * initial


# rl-zoo3 Optuna-tuned Ant-v4 uses a constant (very low) LR; our default uses a
# linear schedule. Resume only restores the schedule cleanly for constant LRs.
PROFILES = {
    "default": dict(n_steps=2048, batch_size=64, n_epochs=10,
                    learning_rate=linear_schedule(3e-4), clip_range=0.2,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5,
                    max_grad_norm=0.5),
    "tuned_ant": dict(n_steps=512, batch_size=32, n_epochs=10,
                      learning_rate=1.90609e-05, clip_range=0.1, gamma=0.98,
                      gae_lambda=0.8, ent_coef=4.9646e-07, vf_coef=0.677239,
                      max_grad_norm=0.6),
    "tuned_walker": dict(n_steps=512, batch_size=32, n_epochs=20,
                         learning_rate=5.05041e-05, clip_range=0.1, gamma=0.99,
                         gae_lambda=0.95, ent_coef=0.000585045, vf_coef=0.871923,
                         max_grad_norm=1.0),
}


class SaveVecNormalize(BaseCallback):
    """Persist VecNormalize stats whenever EvalCallback finds a new best model."""

    def __init__(self, vec_env, path):
        super().__init__()
        self.vec_env, self.path = vec_env, path

    def _on_step(self) -> bool:
        self.vec_env.save(str(self.path))
        return True


def _ckpt_steps(p):
    m = re.search(r"_(\d+)_steps", p.name)
    return int(m.group(1)) if m else -1


def latest_checkpoint():
    ckpts = list(MODEL_DIR.glob("ppo_checkpoint_*_steps.zip"))
    return max(ckpts, key=_ckpt_steps) if ckpts else None


# --- Build envs ---
base_env = envs.make_vec(ENV_ID, n_envs=N_ENVS, seed=0)
eval_env = envs.make_vec(ENV_ID, n_envs=1, seed=99)
if NORMALIZE:
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

# --- Fresh model or resume from the latest checkpoint ---
ckpt = latest_checkpoint() if RESUME else None
if ckpt is not None:
    steps = _ckpt_steps(ckpt)
    if NORMALIZE:
        vn = MODEL_DIR / f"ppo_checkpoint_vecnormalize_{steps}_steps.pkl"
        vec_env = (VecNormalize.load(str(vn), base_env) if vn.exists()
                   else VecNormalize(base_env, norm_obs=True, norm_reward=True,
                                     clip_obs=10.0))
        vec_env.training = True
    else:
        vec_env = base_env
    model = PPO.load(str(ckpt), env=vec_env, device="cpu",
                     tensorboard_log=str(LOG_DIR))
    remaining = max(TOTAL_TIMESTEPS - model.num_timesteps, 0)
    print(f"[expert] RESUME from {ckpt.name} at {model.num_timesteps:,} steps, "
          f"{remaining:,} remaining", flush=True)
    reset_timesteps = False
else:
    vec_env = base_env
    if NORMALIZE:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0)
    model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=str(LOG_DIR),
                seed=0, device="cpu", policy_kwargs=dict(net_arch=config.NET_ARCH),
                **PROFILES[PROFILE])
    remaining = TOTAL_TIMESTEPS
    reset_timesteps = True

# --- Callbacks ---
on_best = SaveVecNormalize(vec_env, VECNORM_PATH) if NORMALIZE else None
eval_cb = EvalCallback(
    eval_env, best_model_save_path=str(MODEL_DIR), log_path=str(LOG_DIR),
    eval_freq=max(20_000 // N_ENVS, 1), n_eval_episodes=10,
    deterministic=True, render=False, callback_on_new_best=on_best)
ckpt_cb = CheckpointCallback(
    save_freq=max(100_000 // N_ENVS, 1), save_path=str(MODEL_DIR),
    name_prefix="ppo_checkpoint", save_vecnormalize=NORMALIZE)

# --- Train ---
model.learn(total_timesteps=remaining, callback=[eval_cb, ckpt_cb],
            tb_log_name="PPO_expert", progress_bar=False,
            reset_num_timesteps=reset_timesteps)
model.save(MODEL_DIR / "ppo_expert_final")

# --- Final evaluation on raw returns ---
if NORMALIZE:
    ev = envs.make_vec(ENV_ID, n_envs=1, seed=123)
    ev = VecNormalize.load(str(VECNORM_PATH), ev)
    ev.training, ev.norm_reward = False, False
    mean, std = evaluate_policy(PPO.load(MODEL_DIR / "best_model"), ev,
                                n_eval_episodes=20, deterministic=True)
else:
    mean, std = eval.evaluate(PPO.load(MODEL_DIR / "best_model"), ENV_ID)

target = config.RETURN_TARGETS[ENV_ID]
print(f"[expert] DONE in {(time.time()-t0)/60:.1f} min | best_model return "
      f"{mean:.1f} +/- {std:.1f} | target {target} -> "
      f"{'PASS' if mean > target else 'NOT YET'}", flush=True)
