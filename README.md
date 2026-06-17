# RL Group Project: PPO + Imitation Learning (Walker2d / Ant)

**Group members:** Marco De Palma, Em Echeverria, Leah Sarouphin,
Juan Jose Rincon Briceño, Matteo Mainetti

A complete two-phase reinforcement learning pipeline in MuJoCo: train a PPO
expert, distill it with Behavioural Cloning and DAgger, and test imitation as a
PPO-pretraining warm start. Run on both `Walker2d-v4` and `Ant-v4`.

The plan, the decisions and why they changed, the final results, and the
answers to the research questions (RQ1-RQ6) are in
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Headline results

|                              | Walker2d   | Ant        |
| ---------------------------- | ---------- | ---------- |
| PPO expert (eval return)     | ~6043      | ~6293      |
| Library BC (% of expert)     | 5719 (95%) | 6237 (99%) |
| DAgger (fair, 12 iters)      | 6208       | 6564       |
| PPO from scratch @1.5M steps | ~1126      | ~4965      |
| BC/DAgger + PPO @1.5M steps  | ~5700      | ~6600-6900 |

**Central finding:** imitation pretraining sharply reduces PPO's sample
complexity, BC and DAgger warm-starts reach near-expert return at a fraction of
the from-scratch budget. Full discussion (RQ1-RQ6) is in
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Extended requirements (E1, E2; bonus)

Two from-scratch BC ablations, 5 seeds on both environments (`noise_sweep.py`,
`norm_ablation.py`; figures in [notebook 06](notebooks/06_extended.ipynb)):

| Bonus                           | Walker2d            | Ant                   |
| ------------------------------- | ------------------- | --------------------- |
| E1 collapse (below half-expert) | sigma >= 0.05       | none up to sigma=0.8  |
| E2 normalised vs raw obs        | 4654 vs 1163 (4.0x) | 5679 vs 5946 (~1x)    |

- **E1, noisy expert:** Walker2d is fragile (BC below half the expert by
  sigma=0.05, down to 5% at sigma=0.8); Ant is robust (95-102% of the expert
  through sigma=0.4).
- **E2, observation normalisation:** decisive on Walker2d (a 4.0x gap, raw-obs BC
  fails) but irrelevant on Ant.

Both reinforce RQ6: imitation difficulty, not state dimensionality, governs
sensitivity. A further bonus (off-policy SAC vs on-policy PPO) is set up in
`train_sac.py` and the Colab runner. Full numbers in
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Repository layout

```
notebooks/   01..06  submission notebooks (executed, figures embedded)
src/         shared modules: config, seeding, envs, collect, bc_scratch,
             bc_bridge, dagger, eval, plotting
train_expert.py  collect_demos.py  bc_experiments.py  arch_sweep.py
dagger_run.py    pretraining.py    make_video.py        run scripts (one per stage)
noise_sweep.py   norm_ablation.py                       extended requirements (E1, E2)
colab/colab_runner.ipynb   one-click Colab pipeline
requirements.txt   PROJECT_OVERVIEW.md
models/ data/ outputs/ videos/ logs/   artifacts (git-ignored; in the submission zip)
```

## Environment setup (from scratch)

Requires **Python 3.11** (torch / SB3 / imitation lack reliable 3.13+ wheels).
Using [uv](https://docs.astral.sh/uv/) (or swap for `python -m venv`):

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt        # or: .venv/bin/pip install -r requirements.txt
```

Sanity check:

```bash
.venv/bin/python -c "import gymnasium as gym; gym.make('Walker2d-v4').reset(seed=0); print('ok')"
```

MuJoCo is bundled with `gymnasium[mujoco]`; no licence needed.

## Reproduce from scratch

Run in order. Runtimes are on an Apple Silicon laptop CPU (Colab CPU is slower;
PPO is CPU-bound so a GPU does not speed up expert training). All randomness is
seeded (`src.seeding.set_seed` plus explicit `seed=` to SB3 and `env.reset(seed=)`).

```bash
# 1. PPO experts (M1)            Walker2d ~30-40 min | Ant ~90 min
python train_expert.py Walker2d-v4 8000000 4 norm default
python train_expert.py Ant-v4     10000000 1 norm tuned_ant

# 2. Demonstrations + EDA (M2)   ~1-2 min each
python collect_demos.py Walker2d-v4 100
python collect_demos.py Ant-v4     100

# 3. BC: library+scratch, epoch sweep, ablation (5x5 seeds), arch sweep
#    (M3/M4/M5/M8)               ~40 min per config
python bc_experiments.py Walker2d-v4
python arch_sweep.py     Walker2d-v4
python bc_experiments.py Ant-v4
python arch_sweep.py     Ant-v4

# 4. DAgger (M7)                 ~6-10 min each
python dagger_run.py Walker2d-v4 Walker2d-v4
python dagger_run.py Ant-v4      Ant-v4

# 5. Imitation-as-PPO pretraining (Stage 5)   ~20-40 min per env
python pretraining.py Walker2d-v4 Walker2d-v4 default   1500000 4
python pretraining.py Ant-v4      Ant-v4      tuned_ant 1500000 1

# 6. Side-by-side videos (M6)    ~1 min each
python make_video.py Walker2d-v4 Walker2d-v4
python make_video.py Ant-v4      Ant-v4

# 7. Extended requirements (E1 noisy expert, E2 obs normalisation)  ~15-30 min per env
python noise_sweep.py   Walker2d-v4
python norm_ablation.py Walker2d-v4
python noise_sweep.py   Ant-v4
python norm_ablation.py Ant-v4

# 8. Bonus: SAC off-policy expert (vs on-policy PPO sample efficiency).
#    SAC is update-bound, so a GPU helps; device="auto" picks CUDA when present.
#    No VecNormalize (off-policy replay buffer). Resumable via the latest checkpoint.
python train_sac.py Ant-v4        3000000 tuned_ant         # in-brief comparison
python train_sac.py HalfCheetah-v4 3000000 tuned_halfcheetah # off-brief, targets 8000
```

`train_expert.py` writes `models/ppo_expert_<env>/best_model` (+ `vecnormalize.pkl`)
and checkpoints; it resumes from the last checkpoint if re-run with `resume`
appended. Every stage writes its figures/JSON to `outputs/`.

To also study the second (non-normalised) Walker2d expert used in the report's
both-experts comparison, pass `Walker2d-v4_generic_backup` as the `DATA_KEY`
(second argument) to `bc_experiments.py` / `arch_sweep.py` / `dagger_run.py`.

## Notebooks (the submission)

The six notebooks run **top-to-bottom with no manual intervention** and are
crash-guarded (missing artifacts print a hint rather than erroring). They set all
seeds, load the submitted expert checkpoints, and:

- **re-run the light stages live** (expert evaluation, demonstration collection,
  BC training, a short DAgger) so the results are genuinely reproduced in minutes;
- **display precomputed figures** for the heavy multi-seed sweeps and the PPO
  pretraining (produced by the scripts above), which take too long to re-run inline.

| Notebook           | Deliverables   | Research questions         |
| ------------------ | -------------- | -------------------------- |
| 01_ppo_expert      | M1             | hyperparameter rationale   |
| 02_data_collection | M2             | dataset quality, EDA       |
| 03_bc_student      | M3, M4, M5, M8 | RQ1-RQ4                    |
| 04_dagger          | M7             | RQ5                        |
| 05_pretraining     | Stage 5        | RQ6 + consolidated RQ1-RQ6 |
| 06_extended        | E1, E2 (bonus) | noise robustness, obs norm |

Launch with `.venv/bin/jupyter notebook` (or open in VS Code / Colab).

## Running on Colab

Open [colab/colab_runner.ipynb](colab/colab_runner.ipynb) in Colab (File ->
Open notebook -> GitHub -> `em-ech/rl-ppo-imitation-learning`). It clones the
repo, installs deps, and runs each stage with artifacts persisted to Drive. Note:
on Colab the numpy 1.26 pin requires a one-time runtime restart after install
(the notebook says when), and the dependency-conflict warnings about
jax/transformers/opencv are harmless (those packages are unused here).

## Reproducibility notes

- Seeds: `numpy`, `torch`, Python `random` (via `src.seeding.set_seed`), the
  environments (`reset(seed=)`), and SB3 (`seed=`) are all set.
- The experts use `VecNormalize`; BC/DAgger train and evaluate on the same
  normalised observations (handled in `src.eval`).
- Notebooks are crash-guarded: a fresh run without precomputed artifacts prints
  guidance instead of failing.
