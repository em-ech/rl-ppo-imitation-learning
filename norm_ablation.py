"""Observation-normalisation ablation (E2). Runs on MPS.

Extended requirement E2: train BC with and without normalising observations to
zero mean / unit variance, and compare convergence speed and final performance.
To isolate the effect of input normalisation, both conditions learn the same
recorded (raw-obs -> action) mapping; the "normalised" condition standardises the
observations with the demonstration set's own mean/std (the standardiser is reused
at evaluation), while the "raw" condition feeds unscaled observations. This is the
controlled version of the normalisation finding noted in PROJECT_OVERVIEW.
Final return uses config.SEEDS for error bars; the seed-0 validation-loss curves
drive the convergence panel.
Usage: .venv/bin/python norm_ablation.py [ENV_ID]
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
seeding.set_seed(0)

data = collect.load(config.DATA_DIR / DATA_KEY)
obs, acts = data["observations"], data["actions"]
expert_mean = float(data["episode_returns"].mean())
t0 = time.time()
print(f"[norm] env={ENV_ID} device={DEVICE} seeds={config.SEEDS}", flush=True)


def standardiser(hist):
    """Build an obs -> standardised-obs fn from train_bc's fitted mean/std."""
    mean, std = hist["obs_mean"][0], hist["obs_std"][0]
    return lambda o: ((np.asarray(o, dtype=np.float32) - mean) / std).astype(np.float32)


conditions = {"normalised": True, "raw": False}
# Resumable: each condition is checkpointed to the JSON when it completes, and a
# re-run skips conditions already present, so a shutdown never loses finished work.
JSON_PATH = config.OUTPUTS_DIR / f"bc_norm_ablation_{DATA_KEY}.json"
if JSON_PATH.exists():
    with open(JSON_PATH) as f:
        results = json.load(f)
    results["condition"] = results.get("condition", {})
    print(f"[norm] resuming: {sorted(results['condition'])} already done", flush=True)
else:
    results = {"env": ENV_ID, "data_key": DATA_KEY, "expert_mean": expert_mean,
               "seeds": config.SEEDS, "n_eval_episodes": 20, "condition": {}}

for label, do_norm in conditions.items():
    if label in results["condition"]:
        continue
    runs, curve = [], None
    for seed in config.SEEDS:
        student, hist = bc_scratch.train_bc(obs, acts, seed=seed, n_epochs=EPOCHS,
                                            batch_size=BATCH, lr=LR, device=DEVICE,
                                            normalize=do_norm)
        norm = standardiser(hist) if do_norm else None
        m, _ = eval.evaluate_torch(student, ENV_ID, DEVICE, n_episodes=20, normalizer=norm)
        runs.append(float(m))
        if seed == 0:
            curve = hist["val"]
    mean_r, std_r = float(np.mean(runs)), float(np.std(runs))
    results["condition"][label] = {"mean": mean_r, "std": std_r, "raw": runs,
                                   "val_curve": curve, "epochs_to_converge": len(curve)}
    with open(JSON_PATH, "w") as f:  # checkpoint after each condition
        json.dump(results, f, indent=2)
    print(f"[norm] {label:11s}: {mean_r:.1f} +/- {std_r:.1f}  "
          f"converged@{len(curve)}ep  raw={[round(r) for r in runs]}  "
          f"({time.time()-t0:.0f}s)", flush=True)

if all(c in results["condition"] for c in conditions):
    curves = {c: results["condition"][c]["val_curve"] for c in conditions}
    finals = {c: (results["condition"][c]["mean"], results["condition"][c]["std"])
              for c in conditions}
    plotting.save(plotting.norm_ablation(curves, finals, expert_mean,
                  title=f"Observation Normalisation Ablation ({DATA_KEY}, E2)"),
                  config.OUTPUTS_DIR / f"bc_norm_ablation_{DATA_KEY}.png")
    print(f"[norm] DONE in {(time.time()-t0)/60:.1f} min -> {JSON_PATH}", flush=True)
