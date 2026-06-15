"""DAgger core, shared by dagger_run.py and notebook 04 (DRY).

Reimplements the brief's broken listing against imitation 1.0.0: a real BC
trainer (the listing passed None), the expert's .policy as oracle, and the
expert's VecNormalize wrapped on the DAgger venv so the student and the queried
expert both see normalised observations. CPU-only (imitation is not GPU-safe).
"""
from __future__ import annotations

import tempfile

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from . import config, eval


def run_dagger(env_id: str, data_key: str | None = None, n_iters: int = 12,
               steps_per_iter: int = 5000, bc_epochs: int = 25, seed: int = 0,
               eval_episodes: int = 10, verbose: bool = True):
    """Run DAgger from the saved expert. Returns a dict with the per-iteration
    returns, dataset sizes, the trained policy, and the expert mean.
    """
    from imitation.algorithms import bc
    from imitation.algorithms.dagger import SimpleDAggerTrainer

    data_key = data_key or env_id
    model_dir = config.MODELS_DIR / f"ppo_expert_{data_key}"
    vn_path = model_dir / "vecnormalize.pkl"
    rng = np.random.default_rng(seed)
    expert = PPO.load(model_dir / "best_model", device="cpu")

    def make_venv():
        venv = DummyVecEnv([lambda: gym.make(env_id)])
        if vn_path.exists():
            venv = VecNormalize.load(str(vn_path), venv)
            venv.training, venv.norm_reward = False, False
        return venv

    venv = make_venv()
    student_policy = PPO("MlpPolicy", venv, seed=seed, device="cpu",
                         policy_kwargs=dict(net_arch=config.NET_ARCH)).policy
    bc_trainer = bc.BC(observation_space=venv.observation_space,
                       action_space=venv.action_space, rng=rng,
                       policy=student_policy, batch_size=256,
                       optimizer_kwargs={"lr": 1e-4}, device="cpu")

    returns_by_iter, dataset_sizes = [], []
    with tempfile.TemporaryDirectory(prefix="dagger_") as tmp:
        trainer = SimpleDAggerTrainer(venv=venv, scratch_dir=tmp,
                                      expert_policy=expert.policy, rng=rng,
                                      bc_trainer=bc_trainer)
        for i in range(n_iters):
            trainer.train(total_timesteps=steps_per_iter,
                          bc_train_kwargs={"n_epochs": bc_epochs,
                                           "progress_bar": False})
            m, _ = eval.evaluate(trainer.policy, env_id, n_episodes=eval_episodes,
                                 vecnorm_path=vn_path if vn_path.exists() else None)
            returns_by_iter.append(float(m))
            dataset_sizes.append((i + 1) * steps_per_iter)
            if verbose:
                print(f"  DAgger iter {i+1:2d}/{n_iters}: return {m:.1f}", flush=True)

    return {"returns_by_iter": returns_by_iter, "dataset_sizes": dataset_sizes,
            "policy": trainer.policy, "vecnorm_path": vn_path}
