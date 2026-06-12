"""Full PPO expert training (spec Section 3, M1).

Mirrors notebooks/01_ppo_expert.ipynb so the long run can go in the background.
Usage: .venv/bin/python train_expert.py [ENV_ID] [TOTAL_TIMESTEPS] [N_ENVS] [norm]

The optional 4th arg ('norm') enables VecNormalize observation normalisation,
which is the standard fix for PPO on Ant. When enabled, the running statistics
are saved next to best_model as vecnormalize.pkl so downstream collection and
evaluation can reuse them.
"""
import sys
import time

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

seeding.set_seed(0)
MODEL_DIR = config.MODELS_DIR / f"ppo_expert_{ENV_ID}"
LOG_DIR = config.LOGS_DIR / f"ppo_expert_{ENV_ID}"
VECNORM_PATH = MODEL_DIR / "vecnormalize.pkl"
t0 = time.time()
print(f"[expert] env={ENV_ID} steps={TOTAL_TIMESTEPS:,} n_envs={N_ENVS} "
      f"normalize={NORMALIZE} -> {MODEL_DIR}", flush=True)


def linear_schedule(initial: float):
    """LR decays linearly from `initial` to 0 over training (progress 1 -> 0)."""
    return lambda progress_remaining: progress_remaining * initial


class SaveVecNormalize(BaseCallback):
    """Persist VecNormalize stats whenever EvalCallback finds a new best model."""

    def __init__(self, vec_env, path):
        super().__init__()
        self.vec_env, self.path = vec_env, path

    def _on_step(self) -> bool:
        self.vec_env.save(str(self.path))
        return True


vec_env = envs.make_vec(ENV_ID, n_envs=N_ENVS, seed=0)
eval_env = envs.make_vec(ENV_ID, n_envs=1, seed=99)
if NORMALIZE:
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

on_best = SaveVecNormalize(vec_env, VECNORM_PATH) if NORMALIZE else None
eval_cb = EvalCallback(
    eval_env, best_model_save_path=str(MODEL_DIR), log_path=str(LOG_DIR),
    eval_freq=max(20_000 // N_ENVS, 1), n_eval_episodes=10,
    deterministic=True, render=False, callback_on_new_best=on_best)
ckpt_cb = CheckpointCallback(
    save_freq=max(100_000 // N_ENVS, 1), save_path=str(MODEL_DIR),
    name_prefix="ppo_checkpoint")

model = PPO(
    "MlpPolicy", vec_env, n_steps=2048, batch_size=64, n_epochs=10,
    learning_rate=linear_schedule(3e-4), clip_range=0.2, gamma=0.99,
    gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, verbose=0,
    tensorboard_log=str(LOG_DIR), seed=0, device="cpu",
    policy_kwargs=dict(net_arch=config.NET_ARCH))

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, ckpt_cb],
            tb_log_name="PPO_expert", progress_bar=False)
model.save(MODEL_DIR / "ppo_expert_final")

# Final evaluation on raw returns.
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
