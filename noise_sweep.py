"""Noisy-expert BC sweep (E1). Runs on MPS.

Extended requirement E1: add zero-mean Gaussian noise to the recorded expert
actions at increasing std and observe how robustly the student learns, locating
the noise level at which performance collapses. Each std is trained across
config.SEEDS and evaluated, so the curve has proper error bars. Observations use
the expert's VecNormalize representation (as in the main BC pipeline); noise is
added only to the action targets, then clipped to the valid [-1, 1] torque range.
Usage: .venv/bin/python noise_sweep.py [ENV_ID]
"""
import json
import sys
import time

import numpy as np

from src import bc_scratch, collect, config, eval, plotting, seeding

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
DATA_KEY = sys.argv[2] if len(sys.argv) > 2 else ENV_ID
DEVICE = config.device()
EPOCHS, LR, BATCH = 150, 1e-4, 256  # ceiling; bc_scratch early-stops
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
seeding.set_seed(0)

data = collect.load(config.DATA_DIR / DATA_KEY)
obs, acts = data["observations"], data["actions"]
expert_mean = float(data["episode_returns"].mean())
VN_PATH = config.MODELS_DIR / f"ppo_expert_{DATA_KEY}" / "vecnormalize.pkl"
norm = eval.load_obs_normalizer(VN_PATH)
N = (lambda x: norm(x)) if norm is not None else (lambda x: x)
obs_n = N(obs)
t0 = time.time()
print(f"[noise] env={ENV_ID} device={DEVICE} seeds={config.SEEDS} "
      f"sigmas={SIGMAS} normalized={norm is not None}", flush=True)

# Resumable: per-sigma results are checkpointed to the JSON after each level, and
# a re-run skips any sigma already present, so a shutdown mid-sweep never loses or
# recomputes finished work.
JSON_PATH = config.OUTPUTS_DIR / f"bc_noise_sweep_{DATA_KEY}.json"
if JSON_PATH.exists():
    with open(JSON_PATH) as f:
        results = json.load(f)
    results["noise"] = results.get("noise", {})
    print(f"[noise] resuming: {sorted(results['noise'])} already done", flush=True)
else:
    results = {"env": ENV_ID, "data_key": DATA_KEY, "expert_mean": expert_mean,
               "seeds": config.SEEDS, "n_eval_episodes": 20,
               "normalized": norm is not None, "sigmas": SIGMAS, "noise": {}}

for sigma in SIGMAS:
    if str(sigma) in results["noise"]:
        continue
    runs = []
    for seed in config.SEEDS:
        rng = np.random.default_rng(seed)
        noisy = acts if sigma == 0.0 else np.clip(
            acts + rng.normal(0.0, sigma, acts.shape).astype(np.float32), -1.0, 1.0)
        student, _ = bc_scratch.train_bc(obs_n, noisy, seed=seed, n_epochs=EPOCHS,
                                         batch_size=BATCH, lr=LR, device=DEVICE)
        m, _ = eval.evaluate_torch(student, ENV_ID, DEVICE, n_episodes=20, normalizer=norm)
        runs.append(float(m))
    results["noise"][str(sigma)] = {"mean": float(np.mean(runs)),
                                    "std": float(np.std(runs)), "raw": runs}
    with open(JSON_PATH, "w") as f:  # checkpoint after each sigma
        json.dump(results, f, indent=2)
    print(f"[noise] sigma={sigma:<4}: {np.mean(runs):.1f} +/- {np.std(runs):.1f}  "
          f"raw={[round(r) for r in runs]}  ({time.time()-t0:.0f}s)", flush=True)

if all(str(s) in results["noise"] for s in SIGMAS):
    means = [results["noise"][str(s)]["mean"] for s in SIGMAS]
    stds = [results["noise"][str(s)]["std"] for s in SIGMAS]
    plotting.save(plotting.noise_curve(SIGMAS, means, stds, expert_mean,
                  title=f"BC Robustness to Noisy Expert Actions ({DATA_KEY}, E1)"),
                  config.OUTPUTS_DIR / f"bc_noise_sweep_{DATA_KEY}.png")
    print(f"[noise] DONE in {(time.time()-t0)/60:.1f} min -> {JSON_PATH}", flush=True)
