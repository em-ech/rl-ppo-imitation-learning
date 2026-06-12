"""Generate the five notebook stubs as valid nbformat v4 JSON.

Run once: .venv/bin/python scripts_build_notebooks.py
Each notebook wires into src/ and marks the requirements it satisfies.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)

SETUP = (
    "# Make src importable whether run from notebooks/ or project root\n"
    "import sys, os\n"
    "from pathlib import Path\n"
    "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
    "sys.path.insert(0, str(ROOT))\n"
    "# On Colab: mount Drive and set PROJECT_DATA_ROOT before importing src\n"
    "from src import config, seeding, envs, collect, eval, plotting\n"
    "seeding.set_seed(0)\n"
    "ENV_ID = 'Walker2d-v4'  # switch to 'Ant-v4' for the second environment\n"
    "print('device', config.device(), '| env', ENV_ID)"
)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "01_ppo_expert.ipynb": [
        md("# 01 - PPO Expert Training (M1, RQ1)\n\n"
           "**Group members:** TODO\n\n"
           "Train a PPO expert on `Walker2d-v4` and `Ant-v4` to the spec thresholds "
           "(Walker2d > 3000, Ant > 4000). Deliverable: TensorBoard return curves, "
           "best checkpoint, hyperparameter discussion."),
        code(SETUP),
        md("## Training configuration\n\n"
           "Defaults follow the spec's Table 1. For the hyperparameter pilots, run "
           "short budgets (e.g. 500k steps) over gamma, gae_lambda, lr schedule, "
           "n_steps, ent_coef; lock the best, then run the full budget. Document each "
           "choice and why (M1 requirement)."),
        code("from stable_baselines3 import PPO\n"
             "from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback\n\n"
             "MODEL_DIR = config.MODELS_DIR / f'ppo_expert_{ENV_ID}'\n"
             "LOG_DIR   = config.LOGS_DIR / f'ppo_expert_{ENV_ID}'\n"
             "N_ENVS = 4\n"
             "TOTAL_TIMESTEPS = 2_000_000  # Walker2d; raise for Ant. Use ~50_000 for a pilot.\n\n"
             "vec_env  = envs.make_vec(ENV_ID, n_envs=N_ENVS, seed=0)\n"
             "eval_env = envs.make_vec(ENV_ID, n_envs=1, seed=99)\n\n"
             "# NOTE: EvalCallback.eval_freq counts rollout steps, not env steps, so\n"
             "# divide by N_ENVS to evaluate every ~20k environment steps.\n"
             "eval_callback = EvalCallback(\n"
             "    eval_env, best_model_save_path=str(MODEL_DIR), log_path=str(LOG_DIR),\n"
             "    eval_freq=max(20_000 // N_ENVS, 1), n_eval_episodes=10,\n"
             "    deterministic=True, render=False)\n"
             "checkpoint_callback = CheckpointCallback(\n"
             "    save_freq=max(100_000 // N_ENVS, 1), save_path=str(MODEL_DIR),\n"
             "    name_prefix='ppo_checkpoint')"),
        code("model = PPO(\n"
             "    'MlpPolicy', vec_env, n_steps=2048, batch_size=64, n_epochs=10,\n"
             "    learning_rate=3e-4, clip_range=0.2, gamma=0.99, gae_lambda=0.95,\n"
             "    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, verbose=1,\n"
             "    tensorboard_log=str(LOG_DIR), seed=0,\n"
             "    policy_kwargs=dict(net_arch=config.NET_ARCH))\n\n"
             "model.learn(total_timesteps=TOTAL_TIMESTEPS,\n"
             "            callback=[eval_callback, checkpoint_callback],\n"
             "            tb_log_name='PPO_expert', progress_bar=True)\n"
             "model.save(MODEL_DIR / 'ppo_expert_final')\n"
             "print('saved best model to', MODEL_DIR)"),
        md("## Evaluate the best checkpoint\n\n"
           "Load `best_model` (saved by EvalCallback, not the final model) and confirm "
           "the threshold before Phase 2. Monitor live with "
           "`tensorboard --logdir logs/` and watch `rollout/ep_rew_mean`."),
        code("mean, std = eval.evaluate(PPO.load(MODEL_DIR / 'best_model'), ENV_ID)\n"
             "print(f'expert mean reward: {mean:.1f} +/- {std:.1f}')\n"
             "print('threshold', config.RETURN_TARGETS[ENV_ID],\n"
             "      '->', 'PASS' if mean > config.RETURN_TARGETS[ENV_ID] else 'NOT YET')"),
    ],
    "02_data_collection.ipynb": [
        md("# 02 - Expert Demonstration Collection (M2)\n\n**Group members:** TODO\n\n"
           "Roll out the deterministic expert, save the dataset (with per-episode "
           "boundaries), and run EDA. Quality gate: >= 90% of episodes above two "
           "thirds of the eval mean."),
        code(SETUP),
        code("from stable_baselines3 import PPO\n"
             "model = PPO.load(config.MODELS_DIR / f'ppo_expert_{ENV_ID}' / 'best_model')\n"
             "out_dir = config.DATA_DIR / ENV_ID\n"
             "data = collect.collect(model, ENV_ID, n_episodes=100, out_dir=out_dir, seed=0)\n"
             "print('episodes', len(data['episode_returns']), '| transitions', len(data['observations']))\n"
             "print('mean return %.1f' % data['episode_returns'].mean())"),
        md("## Exploratory data analysis (spec Section 5.3)"),
        code("fig = plotting.dataset_eda(data['episode_returns'], data['actions'])\n"
             "plotting.save(fig, config.OUTPUTS_DIR / f'dataset_analysis_{ENV_ID}.png')"),
        md("## Quality gate"),
        code("import numpy as np\n"
             "thr = (2/3) * data['episode_returns'].mean()\n"
             "frac = np.mean(data['episode_returns'] > thr)\n"
             "print('fraction above 2/3 mean: %.2f' % frac)\n"
             "assert frac >= 0.9, 'dataset quality gate failed'"),
    ],
    "03_bc_student.ipynb": [
        md("# 03 - Behavioural Cloning (M3, M4, M5, M8, RQ1, RQ3, RQ4)\n\n"
           "**Group members:** TODO\n\n"
           "Library BC vs from-scratch BC, dataset-size ablation, architecture sweep, "
           "and offline-vs-online metric divergence."),
        code(SETUP),
        code("data = collect.load(config.DATA_DIR / ENV_ID)\n"
             "expert_mean = float(data['episode_returns'].mean())\n"
             "DEVICE = config.device()"),
        md("## M3/M4 - Library BC (imitation) vs from-scratch BC\n\n"
           "Hold epochs / lr / batch / seed equal for a fair comparison."),
        code("from src import bc_bridge, bc_scratch\n"
             "# Library version\n"
             "trainer, env = bc_bridge.train_bc_imitation(data['observations'], data['actions'],\n"
             "                                            ENV_ID, n_epochs=50, device=DEVICE)\n"
             "lib_mean, lib_std = eval.evaluate(trainer.policy, ENV_ID)\n"
             "# From-scratch version\n"
             "student, hist = bc_scratch.train_bc(data['observations'], data['actions'],\n"
             "                                    n_epochs=50, device=DEVICE)\n"
             "scr_mean, scr_std = eval.evaluate_torch(student, ENV_ID, DEVICE)\n"
             "print('library  %.1f +/- %.1f' % (lib_mean, lib_std))\n"
             "print('scratch  %.1f +/- %.1f' % (scr_mean, scr_std))\n"
             "plotting.save(plotting.learning_curves(hist['train'], hist['val']),\n"
             "              config.OUTPUTS_DIR / f'bc_learning_curves_{ENV_ID}.png')"),
        md("## M5 - Dataset-size ablation (RQ2)\n\n"
           "{5, 10, 20, 50, 100} episodes x 5 seeds; mean +/- std with error bars."),
        code("import numpy as np\n"
             "episode_counts, results = [5, 10, 20, 50, 100], {}\n"
             "for n_ep in episode_counts:\n"
             "    sub_obs, sub_acts = collect.subset(data, n_ep)\n"
             "    runs = []\n"
             "    for seed in config.SEEDS:\n"
             "        s, _ = bc_scratch.train_bc(sub_obs, sub_acts, seed=seed, n_epochs=50, device=DEVICE)\n"
             "        m, _ = eval.evaluate_torch(s, ENV_ID, DEVICE)\n"
             "        runs.append(m)\n"
             "    results[n_ep] = runs\n"
             "    print('n_ep=%3d mean=%.1f std=%.1f' % (n_ep, np.mean(runs), np.std(runs)))\n"
             "means = [np.mean(results[n]) for n in episode_counts]\n"
             "stds  = [np.std(results[n]) for n in episode_counts]\n"
             "plotting.save(plotting.ablation(episode_counts, means, stds, expert_mean),\n"
             "              config.OUTPUTS_DIR / f'bc_ablation_data_size_{ENV_ID}.png')"),
        md("## M8 - Architecture sweep (RQ4)\n\n"
           "Small MLP, default, large MLP, skip-connections. Does capacity change "
           "the gap to the expert?"),
        code("archs = {'small': (64, 64), 'default': (256, 256), 'large': (512, 512)}\n"
             "for name, hidden in archs.items():\n"
             "    s, _ = bc_scratch.train_bc(data['observations'], data['actions'],\n"
             "                               hidden=hidden, n_epochs=50, device=DEVICE)\n"
             "    m, sd = eval.evaluate_torch(s, ENV_ID, DEVICE)\n"
             "    print('%-8s %.1f +/- %.1f' % (name, m, sd))\n"
             "# skip-connection variant: bc_scratch.train_bc(..., skip=True)"),
        md("## RQ3 - Does lower BC loss imply a better policy?\n\n"
           "Compare validation MSE against environment return across epochs and discuss."),
    ],
    "04_dagger.ipynb": [
        md("# 04 - DAgger (M7, RQ5) + bonus\n\n**Group members:** TODO\n\n"
           "Rewrite the provided DAgger listing against imitation 1.0.0, compare BC vs "
           "DAgger over iterations, and visualise dataset growth. Listing 10 in the spec "
           "is broken (bc_trainer=None, stale API) and must be reimplemented here."),
        code(SETUP),
        code("import numpy as np, tempfile\n"
             "from stable_baselines3 import PPO\n"
             "from imitation.algorithms import bc\n"
             "from imitation.algorithms.dagger import SimpleDAggerTrainer\n"
             "from stable_baselines3.common.vec_env import DummyVecEnv\n\n"
             "expert = PPO.load(config.MODELS_DIR / f'ppo_expert_{ENV_ID}' / 'best_model')\n"
             "rng = np.random.default_rng(0)\n"
             "venv = DummyVecEnv([lambda: __import__('gymnasium').make(ENV_ID)])\n"
             "# TODO: build a real bc.BC trainer, pass it to SimpleDAggerTrainer(bc_trainer=...),\n"
             "#       loop N_ITERATIONS collecting STEPS_PER_ITER, eval each iteration."),
        md("## BC vs DAgger comparison plot"),
        code("# plotting.bc_vs_dagger(bc_returns_by_epoch, dagger_returns_by_iter, expert_mean)"),
        md("## Stage 5 preview - imitation as PPO pretraining\n\n"
           "Use src.bc_bridge.ppo_from_policy to warm-start PPO from the BC/DAgger "
           "policy and compare return vs environment timesteps against PPO-from-scratch "
           "(the project's central sample-efficiency question)."),
    ],
    "05_pretraining.ipynb": [
        md("# 05 - Imitation as PPO Pretraining (Stage 5)\n\n**Group members:** TODO\n\n"
           "Central question: can imitation learning reduce the sample complexity of PPO? "
           "Compare five methods on return vs environment timesteps: PPO from scratch, "
           "BC only, DAgger only, BC + PPO fine-tune, DAgger + PPO fine-tune. Headline "
           "metric is timesteps to reach a target return."),
        code(SETUP),
        code("from src import bc_bridge\n"
             "data = collect.load(config.DATA_DIR / ENV_ID)\n"
             "DEVICE = config.device()\n"
             "# 1) Train BC student (SB3 policy) via imitation\n"
             "trainer, env = bc_bridge.train_bc_imitation(data['observations'], data['actions'],\n"
             "                                            ENV_ID, n_epochs=50, device=DEVICE)\n"
             "# 2) Warm-start PPO from the BC policy and fine-tune\n"
             "vec_env = envs.make_vec(ENV_ID, n_envs=4, seed=0)\n"
             "model = bc_bridge.ppo_from_policy(trainer.policy, vec_env, seed=0, device=DEVICE)\n"
             "# TODO: model.learn(...) with eval logging; compare vs PPO-from-scratch curve"),
    ],
}

for name, cells in NOTEBOOKS.items():
    (NB_DIR / name).write_text(json.dumps(notebook(cells), indent=1))
    print("wrote", name)
