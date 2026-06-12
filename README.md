# RL Group Project: PPO + Imitation Learning (Walker2d / Ant)

Two-phase pipeline: train a PPO expert, then imitate it with Behavioural Cloning
and DAgger, and finally test imitation as PPO pretraining. Full plan in
[PLAN.md](PLAN.md).

## Setup

Requires Python 3.11 (torch / SB3 / imitation lack reliable 3.13+ wheels).

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

Sanity check:

```bash
.venv/bin/python -c "import gymnasium as gym; gym.make('Walker2d-v4').reset(seed=42); print('ok')"
```

## Layout

| Path                                   | Purpose                                                                                               |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `src/`                                 | shared code: env factory, seeding, eval, plotting, collection, BC (scratch + library), BC->PPO bridge |
| `notebooks/01_ppo_expert.ipynb`        | PPO expert training (M1)                                                                              |
| `notebooks/02_data_collection.ipynb`   | demonstration dataset + EDA (M2)                                                                      |
| `notebooks/03_bc_student.ipynb`        | BC library vs scratch, ablation, architecture sweep (M3/M4/M5/M8)                                     |
| `notebooks/04_dagger.ipynb`            | DAgger, BC vs DAgger comparison (M7)                                                                  |
| `notebooks/05_pretraining.ipynb`       | imitation as PPO pretraining (Stage 5)                                                                |
| `models/ data/ outputs/ videos/ logs/` | artifacts (git-ignored)                                                                               |

## Reproducing

Run the notebooks in order 01 to 05. Each sets `ENV_ID` near the top; run once
with `Walker2d-v4` and once with `Ant-v4`. All randomness is seeded via
`src.seeding.set_seed`. Expert training is CPU-bound (MuJoCo sim); from-scratch
BC uses MPS/CUDA if available.

## Running on Colab

Use [colab/colab_runner.ipynb](colab/colab_runner.ipynb), which clones this repo,
installs deps, and runs every stage with artifacts persisted to Drive so
disconnects do not lose work.

1. Open `colab/colab_runner.ipynb` in Colab (open from GitHub via
   File -> Open notebook -> GitHub, or upload the single file).
2. Run the cells top to bottom. Cell 1 clones the code; cell 2 mounts Drive.

PPO training is CPU-bound (the MuJoCo sim is not GPU-accelerated), so a GPU
runtime only speeds up from-scratch BC. Long expert runs need Colab Pro or an
active tab to avoid idle disconnects; checkpoints and Drive persistence allow
recovery either way. The data root is set via `PROJECT_DATA_ROOT`:

```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ['PROJECT_DATA_ROOT'] = '/content/drive/MyDrive/rl_project'
```

## Notes on the spec

- Ant-v4 observation space is 27-dim in gymnasium 0.29 (the spec's 111 is the
  contact-force variant). We use the 27-dim default.
- The provided DAgger listing (Section 6.3) and the Stage 5 fine-tuning step are
  reimplemented here; see [PLAN.md](PLAN.md) section 2 for the full list of fixes.
