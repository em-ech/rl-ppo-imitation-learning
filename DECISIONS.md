# Decisions and Progress Log

The key decisions, why things changed, and the final results. Source material for
the report's methodology and discussion. **The evidence-backed answers to the
research questions (RQ1-RQ6) are in the notebooks**, not here:

- RQ1-RQ4 -> `notebooks/03_bc_student.ipynb`
- RQ5 -> `notebooks/04_dagger.ipynb`
- RQ6 + a consolidated RQ1-RQ6 table -> `notebooks/05_pretraining.ipynb`

---

## Setup and environment

- **Python 3.11 venv (uv)**, not 3.13/3.14: torch / SB3 / imitation lack reliable
  3.13+ wheels.
- **Pinned stack:** stable-baselines3 2.3.2, gymnasium[mujoco] 0.29.1, imitation
  1.0.0, torch 2.5.1, numpy 1.26.4. imitation 1.0 pins gymnasium <1.0 -> gymnasium
  0.29 -> numpy <2. This forces a numpy downgrade + runtime restart on Colab.
- **Dropped `stable-baselines3[extra]`**: it pulls shimmy[atari]/ale-py, which has
  no wheel for Colab's Python and is irrelevant to MuJoCo. Added tqdm+rich for the
  progress bar.
- **Ant-v4 is 27-dim**, not the spec's 111 (that is the contact-force variant).

## Architecture and repo

- **Shared `src/` modules** (config, seeding, envs, collect, bc_scratch, bc_bridge,
  dagger, ppo, eval, plotting) so notebooks/scripts do not duplicate logic.
- **Fixed spec-listing bugs:** deprecated `net_arch=[dict(...)]` -> `dict(...)`;
  the broken DAgger listing (reimplemented in `src/dagger.py`); missing per-episode
  saving in collection (we save `episode_starts` + `episode_lengths`).
- **BC -> PPO bridge** (`bc_bridge.ppo_from_policy`) and the Stage 5 pretraining
  written from scratch (the spec left them without code).
- **Dev tooling** (notebook/colab generators, smoke tests, orchestrators) lives in
  `.claude/dev/` (gitignored), out of the submission.

## PPO experts

- **EvalCallback `eval_freq` divided by n_envs** (it counts rollout steps, not env
  steps).
- **`torch.set_num_threads(1)`**: small-MLP PPO on CPU thrashes with BLAS
  oversubscription (~9 cores, very low throughput); one thread is far faster.
- **Tuned configs are environment-specific (key finding).** The rl-zoo3 Optuna
  profile took Ant from ~2850 to **6293**, but the analogous `tuned_walker` profile
  _failed_ on Walker2d (1640, below the 3000 threshold). Walker2d instead wanted the
  standard config (`n_steps=2048`, `batch=64`, `lr=3e-4` linear) + VecNormalize, 8M
  steps -> **6043**. You cannot transfer one tuned recipe across environments.
- **VecNormalize was the common decisive lever** for reaching high return on both.
- Mistake corrected: an early normalized Ant run was killed during a noisy patch and
  wrongly judged a failure; the full run was fine. Lesson: do not kill long runs on
  early variance.

## Behavioural Cloning: undertraining -> early stopping

- **BC normalisation bug.** A VecNormalize-trained expert consumes normalised
  observations; the BC students initially learned from raw obs with no input
  normalisation and could not fit (Walker2d library BC 1246). Fixed: BC trains and
  evaluates on the expert's normalised observations (`eval.load_obs_normalizer`).
- **Undertraining.** At a fixed 50 epochs the aggressive Walker2d-6043 student
  reached only ~47% of the expert. The epoch sweep showed it kept improving with
  more epochs, so we added **validation-loss early stopping** (spec-recommended;
  150-epoch ceiling) for scratch BC and a 100-epoch budget for library BC. This
  lifted it to ~95%. Takeaway: the "gap" was mostly training budget, not an
  imitability ceiling.

## DAgger

- First run was undertrained (4 BC epochs/iter, Walker2d-6043 best 2402). The **fair
  re-run (12 iters x 25 BC epochs)** matches or beats BC everywhere and reaches
  expert level. The Walker2d-6043 curve climbs ~600 -> ~6200 over iterations as
  on-policy aggregation corrects covariate shift; Ant starts already high. The
  comparison must hold training effort fixed.

## Compute, monitoring, reproducibility

- **Local Mac for PPO** (CPU-bound sim; GPU does not help), Colab for unattended
  bursts (free tier disconnects ~12h).
- **Monitor background runs by disk artifacts, not stdout**: SB3 prints are
  block-buffered in a non-TTY process, so a healthy run can look stalled; checkpoint
  files are the truth. (This caused a chain of phantom diagnoses before we caught it.)
- **Sleep:** `caffeinate` blocks idle but not lid-close sleep; long local runs need
  the lid open or `pmset -c disablesleep 1`.
- **Seeds set everywhere** (numpy, torch, python, env, SB3); notebooks re-run the
  light stages live from the submitted experts and are crash-guarded.

---

## Final results

Experts: Walker2d **6043**, Ant **6293** (both above the spec thresholds).

| Metric                  | Walker2d-6043 | Walker2d-4627 (2nd expert) | Ant-6293      |
| ----------------------- | ------------- | -------------------------- | ------------- |
| Library BC              | 5719 (95%)    | 4591                       | 6237 (99%)    |
| Scratch BC (early stop) | 4238          | 4160                       | 5754          |
| DAgger (fair, 12 iters) | 6208          | 4732                       | 6564          |
| Arch winner (5 seeds)   | large 4982    | ~4500 (tight)              | ~5900 (tight) |

Stage 5 pretraining, eval return at 1.5M env steps (central result):

|          | PPO scratch | BC + PPO | DAgger + PPO |
| -------- | ----------- | -------- | ------------ |
| Walker2d | 1145        | 5690     | 5451         |
| Ant      | 4965        | 6603     | 6881         |

**Imitation pretraining sharply reduces PPO sample complexity** (pretrained reaches
near-expert return where from-scratch lags badly). Nuance: under Ant's tiny LR the
warm-start holds from step 0; under Walker2d's larger LR the warm-started actor dips
early (fresh critic) then recovers fast.

## Remaining to do

- Presentation (Section 11; title slide will list all members).
- Optional: final submission-zip packaging. Bonuses E1/E2 not attempted.
