# RL Group Project: PPO + Imitation Learning (Walker2d / Ant)

**Group members:** Marco De Palma, Em Echeverria, Leah Sarouphin,
Juan Jose Rincon Briceño, Matteo Mainetti

A complete two-phase reinforcement learning pipeline in MuJoCo: train a PPO
expert, distill it with Behavioural Cloning and DAgger, and test imitation as a
PPO-pretraining warm start. Run on both `Walker2d-v4` and `Ant-v4`.

The plan, the decisions and why they changed, the final results, and the
answers to the research questions (RQ1-RQ6) are in
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Comparison of all experiments

The first five rows are the core stages (M1-M7); the rest are bonus studies: the
E1/E2 BC ablations (5 seeds per point, `noise_sweep.py` / `norm_ablation.py`,
figures in [notebook 06](notebooks/06_extended.ipynb)) and the off-policy SAC
experts (`train_sac.py`). Returns are 20-episode deterministic evals.

| Experiment                     | Description                                                       | Walker2d                | Ant                      |
| ------------------------------ | ----------------------------------------------------------------- | ----------------------- | ------------------------ |
| PPO expert                     | the RL expert trained from scratch (oracle for students)          | 6043                    | 6293                     |
| Library BC                     | behavioural cloning of the expert with the `imitation` lib        | 5719 (95%)              | 6237 (99%)               |
| DAgger                         | behavioural cloning with on-policy expert querying                | 6208                    | 6564                     |
| PPO from scratch @1.5M         | PPO trained from a random start, no imitation                     | ~1126                   | ~4965                    |
| BC / DAgger + PPO @1.5M        | PPO warm-started from the imitation policy                        | ~5700                   | ~6600-6900               |
| E1: noisy expert (bonus)       | BC trained on expert actions with added Gaussian noise            | collapses by sigma=0.05 | robust through sigma=0.4 |
| E2: obs normalisation (bonus)  | BC with vs without zero-mean / unit-variance observations         | 4654 vs 1163 (4.0x)     | 5679 vs 5946 (~1x)       |
| SAC expert (off-policy, bonus) | off-policy SAC vs PPO, same env (HalfCheetah-v4 off-brief: 15387) | n/a                     | 7295 @ 3M (vs 6293)      |
| SAC 5M extension (bonus)       | longer SAC run probing Ant's asymptote (stopped at 3.8M)          | n/a                     | ~7200 (no gain over 3M)  |

**Central finding:** imitation pretraining sharply reduces PPO's sample
complexity, BC and DAgger warm-starts reach near-expert return at a fraction of
the from-scratch budget. The two bonus rows reinforce RQ6: Walker2d is fragile to
imperfect imitation (collapses under small action noise, fails without obs
normalisation) while Ant is robust to both, so imitation difficulty, not state
dimensionality, governs sensitivity. A further bonus compares off-policy SAC
against the on-policy PPO experts (see below). Full discussion (RQ1-RQ6) and exact
E1/E2 numbers are in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

Side-by-side renders of every deployable policy (PPO expert, BC, DAgger, SAC) are
in `videos/comparison_<env>.mp4` (`make_comparison_video.py`); the ablation rows
(E1/E2) and the pretraining-stage runs are not single policies, so they are not
rendered.

## Bonus: SAC off-policy experts

We also trained SAC (off-policy, maximum-entropy) and compared it to the
on-policy PPO experts (`train_sac.py`, 20-episode deterministic eval, no
VecNormalize). SAC is markedly more sample-efficient:

| Env                        | PPO expert (on-policy) | SAC (off-policy) |
| -------------------------- | ---------------------- | ---------------- |
| Ant-v4                     | 6293 @ 10M steps       | 7295 @ 3M steps  |
| HalfCheetah-v4 (off-brief) | n/a                    | 15387 @ 3M steps |

SAC beats the PPO Ant expert with over 3x fewer environment steps, and on
HalfCheetah-v4 it clears the 8000 stretch target by a wide margin. A 5M Ant
extension (stopped at 3.8M) confirmed the asymptote: it peaked ~7200 with no gain
over the 3M policy, so 7295 stands as the best Ant SAC return; a sustained 8000 is
above Ant's practical SAC ceiling.

## Repository layout

```
notebooks/   01..06  submission notebooks (executed, figures embedded)
src/         shared modules: config, seeding, envs, collect, bc_scratch,
             bc_bridge, dagger, eval, plotting, sac, video
train_expert.py  collect_demos.py  bc_experiments.py  arch_sweep.py
dagger_run.py    pretraining.py    make_video.py        run scripts (one per stage)
noise_sweep.py   norm_ablation.py                       extended requirements (E1, E2)
train_sac.py     make_comparison_video.py               SAC experts; per-env policy videos
colab/colab_runner.ipynb   full documented pipeline + extended + SAC (Colab)
requirements.txt   PROJECT_OVERVIEW.md
videos/      rendered comparison + expert-vs-student mp4s (tracked in the repo)
models/ data/ outputs/ logs/           artifacts (git-ignored; in the submission zip)
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

Fast utility tests (no MuJoCo/Torch install needed):

```bash
uv run --python 3.11 --with numpy==1.26.4 python -m unittest discover -s tests
```

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

# 6b. Multi-policy comparison videos (PPO/BC/DAgger/SAC side by side per env)
python make_comparison_video.py Walker2d-v4
python make_comparison_video.py Ant-v4
python make_comparison_video.py HalfCheetah-v4

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

[colab/colab_runner.ipynb](colab/colab_runner.ipynb) runs the whole project on
Colab, with a markdown description and the hyperparameter rationale for every
section: PPO experts, demonstrations, BC, DAgger, pretraining, the extended
studies (E1, E2), and the SAC bonus. Open it from GitHub
(`em-ech/rl-ppo-imitation-learning`), set a **GPU**
runtime (helps from-scratch BC and SAC; PPO is CPU-bound and unaffected), and run
top to bottom. Section 1 installs `requirements.txt` and asks for a one-time
runtime restart (the numpy 1.26 pin that `imitation` forces); after restarting,
re-run cells 1.1 and 1.3 and continue. All artifacts persist to Drive via
`PROJECT_DATA_ROOT`, and every long stage resumes from its last checkpoint.

## Reproducibility notes

- Seeds: `numpy`, `torch`, Python `random` (via `src.seeding.set_seed`), the
  environments (`reset(seed=)`), and SB3 (`seed=`) are all set.
- The experts use `VecNormalize`; BC/DAgger train and evaluate on the same
  normalised observations (handled in `src.eval`).
- Notebooks are crash-guarded: a fresh run without precomputed artifacts prints
  guidance instead of failing.
