"""Behavioural Cloning experiments (M3, M4, M5, M8). Mirrors notebook 03.

Runs on MPS for the from-scratch BC so it does not contend with a CPU-bound
expert run. Saves figures, a results JSON, and the BC student weights.
Usage: .venv/bin/python bc_experiments.py [ENV_ID]
"""
import json
import sys
import time

import numpy as np
import torch

from src import bc_bridge, bc_scratch, collect, config, eval, plotting, seeding

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
DEVICE = config.device()
EPOCHS, LR, BATCH = 50, 1e-4, 256
seeding.set_seed(0)

data = collect.load(config.DATA_DIR / ENV_ID)
obs, acts = data["observations"], data["actions"]
expert_mean = float(data["episode_returns"].mean())
results = {"env": ENV_ID, "expert_mean": expert_mean, "device": DEVICE}
t0 = time.time()
print(f"[bc] env={ENV_ID} device={DEVICE} expert_mean={expert_mean:.1f} "
      f"transitions={len(obs):,}", flush=True)

# --- M3/M4: library BC vs from-scratch BC (equal epochs/lr/batch) -------------
# Note: imitation BC minimises the policy's negative log-likelihood (Gaussian),
# while the scratch version minimises pure MSE to the action. Same data and
# budget; the objective differs by construction. We document, not hide, this.
trainer, _ = bc_bridge.train_bc_imitation(obs, acts, ENV_ID, seed=0,
                                          n_epochs=EPOCHS, batch_size=BATCH,
                                          lr=LR, device=DEVICE)
lib_mean, lib_std = eval.evaluate(trainer.policy, ENV_ID)
print(f"[bc] library  BC: {lib_mean:.1f} +/- {lib_std:.1f}", flush=True)

student, hist = bc_scratch.train_bc(obs, acts, seed=0, n_epochs=EPOCHS,
                                    batch_size=BATCH, lr=LR, device=DEVICE)
scr_mean, scr_std = eval.evaluate_torch(student, ENV_ID, DEVICE)
print(f"[bc] scratch  BC: {scr_mean:.1f} +/- {scr_std:.1f} "
      f"(final val MSE {hist['val'][-1]:.4f})", flush=True)

results["library_bc"] = {"mean": lib_mean, "std": lib_std}
results["scratch_bc"] = {"mean": scr_mean, "std": scr_std,
                         "val_mse": hist["val"][-1]}
plotting.save(plotting.learning_curves(hist["train"], hist["val"]),
              config.OUTPUTS_DIR / f"bc_learning_curves_{ENV_ID}.png")

# Save the BC student weights (deliverable models/bc_student.pt)
bc_dir = config.MODELS_DIR / "bc_student"
bc_dir.mkdir(parents=True, exist_ok=True)
torch.save(student.state_dict(), bc_dir / f"bc_student_{ENV_ID}.pt")
trainer.policy.save(str(bc_dir / f"bc_student_imitation_{ENV_ID}"))

# --- M5: dataset-size ablation ({5,10,20,50,100} x 5 seeds) -------------------
episode_counts = [5, 10, 20, 50, 100]
ablation = {}
for n_ep in episode_counts:
    sub_obs, sub_acts = collect.subset(data, n_ep)
    runs = []
    for seed in config.SEEDS:
        s, _ = bc_scratch.train_bc(sub_obs, sub_acts, seed=seed, n_epochs=EPOCHS,
                                   batch_size=BATCH, lr=LR, device=DEVICE)
        m, _ = eval.evaluate_torch(s, ENV_ID, DEVICE, n_episodes=10)
        runs.append(m)
    ablation[n_ep] = runs
    print(f"[bc] ablation n_ep={n_ep:3d}: {np.mean(runs):.1f} +/- {np.std(runs):.1f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
means = [float(np.mean(ablation[n])) for n in episode_counts]
stds = [float(np.std(ablation[n])) for n in episode_counts]
results["ablation"] = {"episode_counts": episode_counts, "means": means,
                       "stds": stds, "raw": ablation}
plotting.save(plotting.ablation(episode_counts, means, stds, expert_mean),
              config.OUTPUTS_DIR / f"bc_ablation_data_size_{ENV_ID}.png")

# --- M8: architecture sweep --------------------------------------------------
archs = {"small (64,64)": dict(hidden=(64, 64)),
         "default (256,256)": dict(hidden=(256, 256)),
         "large (512,512)": dict(hidden=(512, 512)),
         "skip (256,256)": dict(hidden=(256, 256), skip=True)}
results["architecture"] = {}
for name, kw in archs.items():
    s, _ = bc_scratch.train_bc(obs, acts, seed=0, n_epochs=EPOCHS,
                               batch_size=BATCH, lr=LR, device=DEVICE, **kw)
    m, sd = eval.evaluate_torch(s, ENV_ID, DEVICE)
    results["architecture"][name] = {"mean": m, "std": sd}
    print(f"[bc] arch {name:18s}: {m:.1f} +/- {sd:.1f}", flush=True)

with open(config.OUTPUTS_DIR / f"bc_results_{ENV_ID}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[bc] DONE in {(time.time()-t0)/60:.1f} min -> "
      f"outputs/bc_results_{ENV_ID}.json", flush=True)
