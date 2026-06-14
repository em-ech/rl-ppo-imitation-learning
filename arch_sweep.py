"""Multi-seed BC architecture sweep (M8, RQ4). Runs on MPS.

Each architecture is trained across config.SEEDS and evaluated, so the
default/large/skip ordering is reported with proper error bars rather than a
single noisy seed. Usage: .venv/bin/python arch_sweep.py [ENV_ID]
"""
import json
import sys
import time

import numpy as np

from src import bc_scratch, collect, config, eval, plotting, seeding

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v4"
DATA_KEY = sys.argv[2] if len(sys.argv) > 2 else ENV_ID
DEVICE = config.device()
EPOCHS, LR, BATCH = 50, 1e-4, 256
seeding.set_seed(0)

data = collect.load(config.DATA_DIR / DATA_KEY)
obs, acts = data["observations"], data["actions"]
expert_mean = float(data["episode_returns"].mean())
VN_PATH = config.MODELS_DIR / f"ppo_expert_{DATA_KEY}" / "vecnormalize.pkl"
norm = eval.load_obs_normalizer(VN_PATH)
N = (lambda x: norm(x)) if norm is not None else (lambda x: x)
t0 = time.time()
print(f"[arch] env={ENV_ID} device={DEVICE} seeds={config.SEEDS} "
      f"normalized={norm is not None}", flush=True)

archs = {
    "small (64,64)": dict(hidden=(64, 64)),
    "default (256,256)": dict(hidden=(256, 256)),
    "large (512,512)": dict(hidden=(512, 512)),
    "skip (256,256)": dict(hidden=(256, 256), skip=True),
}

results = {"env": ENV_ID, "expert_mean": expert_mean, "seeds": config.SEEDS,
           "n_eval_episodes": 20, "architecture": {}}
for name, kw in archs.items():
    runs = []
    for seed in config.SEEDS:
        s, _ = bc_scratch.train_bc(N(obs), acts, seed=seed, n_epochs=EPOCHS,
                                   batch_size=BATCH, lr=LR, device=DEVICE, **kw)
        m, _ = eval.evaluate_torch(s, ENV_ID, DEVICE, n_episodes=20, normalizer=norm)
        runs.append(float(m))
    results["architecture"][name] = {"mean": float(np.mean(runs)),
                                     "std": float(np.std(runs)), "raw": runs}
    print(f"[arch] {name:18s}: {np.mean(runs):.1f} +/- {np.std(runs):.1f}  "
          f"raw={[round(r) for r in runs]}  ({time.time()-t0:.0f}s)", flush=True)

names = list(archs)
means = [results["architecture"][n]["mean"] for n in names]
stds = [results["architecture"][n]["std"] for n in names]
plotting.save(plotting.arch_bars(names, means, stds, expert_mean,
              title=f"BC Architecture Sweep ({DATA_KEY}, {len(config.SEEDS)} seeds)"),
              config.OUTPUTS_DIR / f"bc_arch_sweep_{DATA_KEY}.png")
with open(config.OUTPUTS_DIR / f"bc_arch_sweep_{DATA_KEY}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[arch] DONE in {(time.time()-t0)/60:.1f} min -> "
      f"outputs/bc_arch_sweep_{DATA_KEY}.json", flush=True)
