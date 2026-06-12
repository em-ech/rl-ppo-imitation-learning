"""End-to-end pipeline validation at tiny scale (NOT a real expert run).

Trains PPO briefly, collects a few demos, runs scratch + library BC, evaluates,
and warm-starts PPO from the BC policy. The goal is to catch integration bugs
across src/ modules before committing to a multi-hour expert run. Returns will
be poor; that is expected.
"""
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from src import bc_bridge, bc_scratch, collect, config, envs, eval, seeding

ENV_ID = "Walker2d-v4"
DEVICE = config.device()
seeding.set_seed(0)
t0 = time.time()
print(f"[pilot] env={ENV_ID} device={DEVICE}")

# 1) Short PPO -----------------------------------------------------------------
MODEL_DIR = config.MODELS_DIR / f"pilot_{ENV_ID}"
N_ENVS = 4
vec_env = envs.make_vec(ENV_ID, n_envs=N_ENVS, seed=0)
eval_env = envs.make_vec(ENV_ID, n_envs=1, seed=99)
eval_cb = EvalCallback(eval_env, best_model_save_path=str(MODEL_DIR),
                       eval_freq=max(5_000 // N_ENVS, 1), n_eval_episodes=3,
                       deterministic=True, render=False)
model = PPO("MlpPolicy", vec_env, seed=0, device="cpu", verbose=0,
            policy_kwargs=dict(net_arch=config.NET_ARCH))
model.learn(total_timesteps=30_000, callback=eval_cb, progress_bar=False)
ppo_mean, ppo_std = eval.evaluate(model, ENV_ID, n_episodes=5)
print(f"[pilot] PPO(30k) return {ppo_mean:.1f} +/- {ppo_std:.1f}  ({time.time()-t0:.0f}s)")

# 2) Collect demos -------------------------------------------------------------
data = collect.collect(model, ENV_ID, n_episodes=5,
                       out_dir=config.DATA_DIR / f"pilot_{ENV_ID}", seed=0)
print(f"[pilot] collected {len(data['episode_returns'])} eps, "
      f"{len(data['observations'])} transitions; "
      f"starts={data['episode_starts'][:5]}")
sub_obs, sub_acts = collect.subset(data, 3)
print(f"[pilot] subset(3) -> {len(sub_obs)} transitions")

# 3) Scratch BC ----------------------------------------------------------------
student, hist = bc_scratch.train_bc(data["observations"], data["actions"],
                                    n_epochs=5, device=DEVICE)
scr_mean, scr_std = eval.evaluate_torch(student, ENV_ID, DEVICE, n_episodes=3)
print(f"[pilot] scratch BC return {scr_mean:.1f} +/- {scr_std:.1f}; "
      f"final val MSE {hist['val'][-1]:.4f}")

# 4) Library BC ----------------------------------------------------------------
trainer, _ = bc_bridge.train_bc_imitation(data["observations"], data["actions"],
                                          ENV_ID, n_epochs=5, device=DEVICE)
lib_mean, lib_std = eval.evaluate(trainer.policy, ENV_ID, n_episodes=3)
print(f"[pilot] library BC return {lib_mean:.1f} +/- {lib_std:.1f}")

# 5) BC -> PPO bridge ----------------------------------------------------------
warm = bc_bridge.ppo_from_policy(trainer.policy, envs.make_vec(ENV_ID, n_envs=2, seed=0),
                                 seed=0, device=DEVICE)
warm.learn(total_timesteps=4_000, progress_bar=False)
warm_mean, _ = eval.evaluate(warm, ENV_ID, n_episodes=3)
print(f"[pilot] BC->PPO warm-start fine-tuned return {warm_mean:.1f}")

print(f"[pilot] FULL PIPELINE OK in {time.time()-t0:.0f}s")
