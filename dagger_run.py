"""DAgger training + BC-vs-DAgger comparison (M7, RQ5). Thin wrapper over
src.dagger.run_dagger (shared with notebook 04). Reimplements the brief's broken
Listing 10 against imitation 1.0.0. CPU-only.
Usage: dagger_run.py [ENV_ID] [DATA_KEY] [N_ITERS] [STEPS_PER_ITER] [BC_EPOCHS]
"""
import json
import sys
import time

import torch

from src import config, dagger, plotting, seeding

torch.set_num_threads(1)
ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
DATA_KEY = sys.argv[2] if len(sys.argv) > 2 else ENV_ID
N_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
STEPS_PER_ITER = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
BC_EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 25
seeding.set_seed(0)
t0 = time.time()
print(f"[dagger] env={ENV_ID} data_key={DATA_KEY} iters={N_ITERS} "
      f"steps/iter={STEPS_PER_ITER} bc_epochs={BC_EPOCHS}", flush=True)

out = dagger.run_dagger(ENV_ID, DATA_KEY, n_iters=N_ITERS,
                        steps_per_iter=STEPS_PER_ITER, bc_epochs=BC_EPOCHS)
returns_by_iter = out["returns_by_iter"]

# Save the student policy.
dagger_dir = config.MODELS_DIR / "dagger_student"
dagger_dir.mkdir(parents=True, exist_ok=True)
out["policy"].save(str(dagger_dir / f"dagger_student_{DATA_KEY}"))

# BC baselines for the comparison plot.
bc_json = config.OUTPUTS_DIR / f"bc_results_{DATA_KEY}.json"
expert_mean, bc_by_epoch = None, None
if bc_json.exists():
    bc_res = json.load(open(bc_json))
    expert_mean = bc_res.get("expert_mean")
    bc_by_epoch = bc_res.get("epoch_sweep", {}).get("returns")
results = {"env": ENV_ID, "data_key": DATA_KEY, "expert_mean": expert_mean,
           "returns_by_iter": returns_by_iter, "dataset_sizes": out["dataset_sizes"]}
with open(config.OUTPUTS_DIR / f"dagger_results_{DATA_KEY}.json", "w") as f:
    json.dump(results, f, indent=2)
if bc_by_epoch and expert_mean:
    plotting.save(plotting.bc_vs_dagger(bc_by_epoch, returns_by_iter, expert_mean),
                  config.OUTPUTS_DIR / f"bc_vs_dagger_{DATA_KEY}.png")

print(f"[dagger] DONE in {(time.time()-t0)/60:.1f} min | final {returns_by_iter[-1]:.1f} "
      f"| best {max(returns_by_iter):.1f} -> outputs/dagger_results_{DATA_KEY}.json",
      flush=True)
