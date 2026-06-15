"""DAgger training and BC-vs-DAgger comparison (M7, RQ5). Reimplements the spec's
broken Listing 10 against imitation 1.0.0.

Wraps the DAgger venv in the expert's VecNormalize stats (when present) so the
queried expert and the student both see normalised observations, matching the BC
pipeline. CPU-only (imitation is not GPU-safe here).
Usage: dagger_run.py [ENV_ID] [DATA_KEY] [N_ITERS] [STEPS_PER_ITER]
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

from src import config, eval, plotting, seeding

torch.set_num_threads(1)
ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
DATA_KEY = sys.argv[2] if len(sys.argv) > 2 else ENV_ID
N_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
STEPS_PER_ITER = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
BC_EPOCHS_PER_ITER = int(sys.argv[5]) if len(sys.argv) > 5 else 25
seeding.set_seed(0)
rng = np.random.default_rng(0)
t0 = time.time()

MODEL_DIR = config.MODELS_DIR / f"ppo_expert_{DATA_KEY}"
VN_PATH = MODEL_DIR / "vecnormalize.pkl"
expert = PPO.load(MODEL_DIR / "best_model", device="cpu")
print(f"[dagger] env={ENV_ID} data_key={DATA_KEY} iters={N_ITERS} "
      f"steps/iter={STEPS_PER_ITER} normalized={VN_PATH.exists()}", flush=True)


def make_venv():
    venv = DummyVecEnv([lambda: gym.make(ENV_ID)])
    if VN_PATH.exists():
        venv = VecNormalize.load(str(VN_PATH), venv)
        venv.training, venv.norm_reward = False, False
    return venv


venv = make_venv()
student_policy = PPO("MlpPolicy", venv, seed=0, device="cpu",
                     policy_kwargs=dict(net_arch=config.NET_ARCH)).policy
bc_trainer = bc.BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    rng=rng, policy=student_policy, batch_size=256,
    optimizer_kwargs={"lr": 1e-4}, device="cpu")

returns_by_iter, dataset_sizes = [], []
with tempfile.TemporaryDirectory(prefix="dagger_") as tmp:
    trainer = SimpleDAggerTrainer(
        venv=venv, scratch_dir=tmp, expert_policy=expert.policy,
        rng=rng, bc_trainer=bc_trainer)
    for i in range(N_ITERS):
        trainer.train(total_timesteps=STEPS_PER_ITER,
                      bc_train_kwargs={"n_epochs": BC_EPOCHS_PER_ITER,
                                       "progress_bar": False})
        m, _ = eval.evaluate(trainer.policy, ENV_ID, n_episodes=10,
                             vecnorm_path=VN_PATH if VN_PATH.exists() else None)
        returns_by_iter.append(float(m))
        dataset_sizes.append((i + 1) * STEPS_PER_ITER)
        print(f"[dagger] iter {i+1:2d}/{N_ITERS} | data~{(i+1)*STEPS_PER_ITER:6d} | "
              f"return {m:.1f}  ({time.time()-t0:.0f}s)", flush=True)

# Save student + results
dagger_dir = config.MODELS_DIR / "dagger_student"
dagger_dir.mkdir(parents=True, exist_ok=True)
trainer.policy.save(str(dagger_dir / f"dagger_student_{DATA_KEY}"))

# Pull BC baselines from the saved BC results for the comparison plot.
bc_json = config.OUTPUTS_DIR / f"bc_results_{DATA_KEY}.json"
expert_mean, bc_by_epoch = None, None
if bc_json.exists():
    bc_res = json.load(open(bc_json))
    expert_mean = bc_res.get("expert_mean")
    bc_by_epoch = bc_res.get("epoch_sweep", {}).get("returns")
results = {"env": ENV_ID, "data_key": DATA_KEY, "expert_mean": expert_mean,
          "returns_by_iter": returns_by_iter, "dataset_sizes": dataset_sizes}
with open(config.OUTPUTS_DIR / f"dagger_results_{DATA_KEY}.json", "w") as f:
    json.dump(results, f, indent=2)

if bc_by_epoch and expert_mean:
    plotting.save(plotting.bc_vs_dagger(bc_by_epoch, returns_by_iter, expert_mean),
                  config.OUTPUTS_DIR / f"bc_vs_dagger_{DATA_KEY}.png")

print(f"[dagger] DONE in {(time.time()-t0)/60:.1f} min | final {returns_by_iter[-1]:.1f} "
      f"| best {max(returns_by_iter):.1f} -> outputs/dagger_results_{DATA_KEY}.json",
      flush=True)
