"""SAC off-policy expert training (bonus experiment: off-policy vs on-policy PPO).

Mirrors train_expert.py but for SAC: single env, replay buffer, automatic entropy
tuning, and NO VecNormalize (running stats would corrupt buffered transitions).
A side effect is that the SAC expert produces raw-observation demonstrations, so
any downstream distillation needs no normalizer. The same EvalCallback protocol
(20 deterministic episodes) and checkpoint/resume mechanism as the PPO expert are
reused, so a multi-hour run is interruptible.

device="auto" lets SB3 use CUDA when present (SAC is update-bound, so a GPU helps,
unlike the sim-bound PPO); it falls back to CPU and avoids MPS op-support pitfalls.

Usage:
  python train_sac.py [ENV_ID] [TOTAL_TIMESTEPS] [PROFILE] [resume]
"""
import re
import sys
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src import config, envs, eval, seeding
from src.sac import SAC_PROFILES

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Ant-v4"
TOTAL_TIMESTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3_000_000
PROFILE = sys.argv[3] if len(sys.argv) > 3 else "tuned_ant"
RESUME = len(sys.argv) > 4 and sys.argv[4].lower() in ("1", "true", "resume")

seeding.set_seed(0)
MODEL_DIR = config.MODELS_DIR / f"sac_expert_{ENV_ID}"
LOG_DIR = config.LOGS_DIR / f"sac_expert_{ENV_ID}"
t0 = time.time()
print(f"[sac] env={ENV_ID} steps={TOTAL_TIMESTEPS:,} profile={PROFILE} "
      f"resume={RESUME} -> {MODEL_DIR}", flush=True)


def _ckpt_steps(p):
    m = re.search(r"_(\d+)_steps", p.name)
    return int(m.group(1)) if m else -1


def latest_checkpoint():
    ckpts = list(MODEL_DIR.glob("sac_checkpoint_*_steps.zip"))
    return max(ckpts, key=_ckpt_steps) if ckpts else None


# --- Build envs (no VecNormalize for off-policy SAC) ---
train_env = envs.make_vec(ENV_ID, n_envs=1, seed=0)
eval_env = envs.make_vec(ENV_ID, n_envs=1, seed=99)

# --- Fresh model or resume from the latest checkpoint ---
ckpt = latest_checkpoint() if RESUME else None
if ckpt is not None:
    model = SAC.load(str(ckpt), env=train_env, device="auto",
                     tensorboard_log=str(LOG_DIR))
    remaining = max(TOTAL_TIMESTEPS - model.num_timesteps, 0)
    print(f"[sac] RESUME from {ckpt.name} at {model.num_timesteps:,} steps, "
          f"{remaining:,} remaining (replay buffer restarts)", flush=True)
    reset_timesteps = False
else:
    model = SAC("MlpPolicy", train_env, verbose=0, tensorboard_log=str(LOG_DIR),
                seed=0, device="auto", **SAC_PROFILES[PROFILE])
    remaining = TOTAL_TIMESTEPS
    reset_timesteps = True

# --- Callbacks (same protocol as the PPO expert) ---
eval_cb = EvalCallback(
    eval_env, best_model_save_path=str(MODEL_DIR), log_path=str(LOG_DIR),
    eval_freq=20_000, n_eval_episodes=10, deterministic=True, render=False)
ckpt_cb = CheckpointCallback(
    save_freq=100_000, save_path=str(MODEL_DIR), name_prefix="sac_checkpoint")

# --- Train ---
model.learn(total_timesteps=remaining, callback=[eval_cb, ckpt_cb],
            tb_log_name="SAC_expert", progress_bar=False,
            reset_num_timesteps=reset_timesteps)
model.save(MODEL_DIR / "sac_expert_final")

# --- Final evaluation on raw returns (no normalisation to undo) ---
mean, std = eval.evaluate(SAC.load(MODEL_DIR / "best_model"), ENV_ID)
target = config.RETURN_TARGETS.get(ENV_ID, 8000.0)
print(f"[sac] DONE in {(time.time()-t0)/60:.1f} min | best_model return "
      f"{mean:.1f} +/- {std:.1f} | target {target} -> "
      f"{'PASS' if mean > target else 'NOT YET'}", flush=True)
