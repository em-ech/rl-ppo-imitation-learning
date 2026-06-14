# Decisions and Progress Log

Chronological record of what we did, the key decisions, and why things changed.
Doubles as source material for the report's methodology and discussion sections.
Newest entries at the bottom. Dates are 2026.

---

## Setup and environment

- **Python 3.11 venv (uv), not 3.13/3.14.** torch / SB3 / imitation lack reliable
  wheels for 3.13+. `uv venv --python 3.11`.
- **Pinned stack:** stable-baselines3 2.3.2, gymnasium[mujoco] 0.29.1, imitation
  1.0.0, torch 2.5.1, numpy 1.26.4.
  - _Why these:_ imitation 1.0.0 pins gymnasium <1.0 -> gymnasium 0.29.1;
    gymnasium 0.29 needs numpy <2 -> numpy 1.26.4. This forces a numpy downgrade
    on Colab (which ships numpy 2.x) and a runtime restart there.
- **Dropped `stable-baselines3[extra]`.** `[extra]` pulls shimmy[atari]/ale-py,
  which has no wheel for Colab's Python and is irrelevant to MuJoCo. Added
  tqdm+rich (the only `[extra]` bits we use, for the progress bar).
- **Ant-v4 is 27-dim, not 111.** The spec's 111 is the contact-force variant; we
  use the gymnasium 0.29 default (27).

## Architecture / repo

- **Shared `src/` modules** (config, seeding, envs, collect, bc_scratch,
  bc_bridge, eval, plotting) so the notebooks/scripts do not duplicate logic.
- **Fixed three spec-listing bugs:** deprecated `net_arch=[dict(...)]` ->
  `net_arch=dict(...)`; the broken DAgger listing (deferred, will reimplement);
  missing per-episode saving in collection (we save `episode_starts` +
  `episode_lengths`, needed by the ablation and the trajectory-length plot).
- **BC -> PPO bridge** (`bc_bridge.ppo_from_policy`) written from scratch; the
  spec left Stage 5 (imitation as PPO pretraining) without code.

## PPO expert training

- **EvalCallback `eval_freq` divided by n_envs.** eval_freq counts rollout steps,
  not env steps; without dividing, evaluation happens n_envs times too rarely.
- **Walker2d expert: linear LR schedule + more steps.** First 2M run hit 3010 but
  with high variance (std 901), borderline for the demonstration quality gate. A
  linear-LR-to-0 schedule plus 4M steps tightened it to ~4616 (std 784). Decision:
  prefer a consistent expert because demonstration quality drives all downstream
  imitation.
- **`torch.set_num_threads(1)`.** Small-MLP PPO on CPU thrashed with BLAS
  oversubscription (~9 cores at very low throughput). One thread runs far faster.

## Ant expert: the hard one

- **Generic config fell short.** Walker2d config (no norm) plateaued ~2422; adding
  VecNormalize and 8M steps reached 2850. The 2850 expert's dataset FAILED the
  quality gate (68% < 90%) because the expert was inconsistent (mean ep length
  763, min return 42).
- **Mistake, corrected:** I killed an early normalized run at 5M during a noisy
  ~1000 patch and wrongly concluded VecNormalize had failed. The full 8M run
  reached 2850, beating the un-normalized 2422. I judged it from an unfinished
  curve. Lesson: do not kill long runs on early variance.
- **SOLVED with the rl-zoo3 Optuna `tuned_ant` profile** (n_envs=1, 1e7 steps,
  normalize, n_steps=512, batch=32, gamma=0.98, lr=1.9e-5, gae_lambda=0.8,
  clip=0.1, vf_coef=0.677): **6293 +/- 1134, target 4000 PASS.** The low-LR tuned
  config was decisive; our generic config could not reach it. So: use `tuned_ant`
  for Ant, `default` (or `tuned_walker`) elsewhere.

## Compute and Colab

- **Local Mac for PPO, Colab for burst.** PPO is CPU-bound (MuJoCo sim), so a GPU
  does not speed up the expert; the Mac is competitive-to-faster than Colab CPU.
  Colab's value is unattended runs, but its free tier disconnects (~12h cap).
- **Colab delivery via a public GitHub repo** (em-ech/rl-ppo-imitation-learning),
  cloned by `colab/colab_runner.ipynb`. Chose this over a Drive zip because the
  user's Colab upload takes only one file (the notebook). Public was the user's
  call despite the coursework-visibility tradeoff.
- **Colab fixes:** numpy-downgrade requires a runtime restart (documented in the
  notebook); the clone cell `cd`s to /content before deleting to survive re-runs;
  cells are working-directory independent.

## Monitoring lesson

- **Judge background runs by disk artifacts, not stdout.** SB3's prints are
  block-buffered in a non-TTY background process and may never reach the log, so
  a healthy run looks stalled. This caused a long chain of phantom diagnoses
  (sleep, then Apple-Silicon efficiency-core demotion). The truth was in the
  checkpoint files landing every ~48s (~2080 steps/s). Our own `[expert] ...`
  prints use `flush=True` and do appear.
- **Sleep:** `caffeinate` blocks idle sleep but not lid-close sleep; `SleepDisabled`
  is not set on this machine. Long local runs need the lid open or
  `sudo pmset -c disablesleep 1`.

## Results so far

| Item                        | Walker2d             | Ant                    |
| --------------------------- | -------------------- | ---------------------- |
| Expert (generic)            | 4616 +/- 784         | 2850 (gate failed)     |
| Expert (tuned profile)      | in progress          | **6293 +/- 1134 PASS** |
| Dataset gate                | 93% PASS             | 90% PASS (tuned)       |
| BC library                  | 4281 (91%)           | 6113 (99%)             |
| BC scratch                  | 3249\* (91% at best) | 5618 (91%)             |
| Arch sweep winner (5 seeds) | large 4055           | skip 5940              |

\* high single-seed variance; multi-seed default ~2429 +/- 1114.

Cross-environment findings (for RQ4/RQ6): BC recovers a _higher_ fraction of the
expert on Ant (99%) than Walker2d (91%) with lower variance, because the tuned Ant
expert gives cleaner full-length demos. Best architecture differs by environment
(large for Walker2d, skip for Ant), so RQ4 is not universal.

## Tuned configs are environment-specific (key finding)

- **`tuned_walker` FAILED: 1640 +/- 686** at 5M, far below our generic config's
  4616 and even the 3000 threshold. The same rl-zoo3 Optuna approach that took Ant
  from 2850 to 6293 made Walker2d *worse*. So you cannot transfer "use the tuned
  config" across environments: Ant wants the low-LR tuned profile, Walker2d wants
  the standard config (n_steps 2048, batch 64, lr 3e-4 linear). Strong report point
  for RQ6 / hyperparameter discussion.
- **Restored** the generic 4616 expert as active (verified 4627 +/- 953); the bad
  run is kept as `models/ppo_expert_Walker2d-v4_tuned1640` for the record.
- Walker2d best remains 4627. The 6000-8000 goal is above the realistic PPO ceiling
  for Walker2d; the untried lever is generic-config + VecNormalize + more steps.

## Repo hygiene

- **Dev tooling moved to `.claude/dev/` (gitignored).** Notebook generators, the
  pilot smoke test, and the overnight orchestrator are not submission deliverables,
  so they stay out of the repo. Run scripts (`train_expert.py`, `collect_demos.py`,
  `bc_experiments.py`, `arch_sweep.py`), `src/`, notebooks, and the Colab runner
  remain because they are part of reproduction. `DECISIONS.md` and `PLAN.md` kept at
  root as project docs.

## Still to do

DAgger (M7), Stage 5 pretraining comparison, M6 side-by-side video, notebook +
presentation assembly. Both environments. Re-collect Walker2d demos and re-run
Walker2d BC once the tuned expert finishes (new expert uses VecNormalize).
