# Project Overview: PPO + Imitation Learning (Walker2d / Ant)

**Group members:** Marco De Palma, Em Echeverria, Leah Sarouphin,
Juan Jose Rincon Briceño, Matteo Mainetti

A two-phase reinforcement-learning pipeline in MuJoCo: train a PPO expert, distill
it with Behavioural Cloning (BC) and DAgger, and test imitation as a PPO-pretraining
warm start, on both `Walker2d-v4` and `Ant-v4`. This is the single source for the
plan, the full decision history (what we ran, why we re-ran it, and why some things
failed), the final results, and the answers to the research questions. Setup and
reproduction commands are in the [README](README.md); the notebooks hold the
runnable experiments and figures.

---

## 1. Experimental plan

Five stages, each building on the previous one:

1. **PPO expert** (`train_expert.py`) - train an expert per environment; the oracle
   for everything downstream.
2. **Demonstrations** (`collect_demos.py`) - roll out the deterministic expert,
   record (obs, action) pairs, run EDA, check a quality gate.
3. **Behavioural Cloning** (`bc_experiments.py`, `arch_sweep.py`) - library and
   from-scratch BC, an epoch sweep, a dataset-size ablation (5 seeds), an
   architecture sweep.
4. **DAgger** (`dagger_run.py`) - on-policy expert querying to fight covariate shift,
   compared against BC.
5. **Imitation as PPO pretraining** (`pretraining.py`) - PPO from scratch vs PPO
   warm-started from BC / DAgger; the central sample-efficiency question.

Plus M6 side-by-side expert-vs-student videos (`make_video.py`).

## 2. Decision history: what we ran, why, and what changed

### Environment and stack

Python 3.11 venv (torch/SB3/imitation lack reliable 3.13+ wheels). The dependency
chain forces specific pins: imitation 1.0 requires gymnasium <1.0 -> gymnasium 0.29
-> numpy <2 -> numpy 1.26. On Colab (which ships numpy 2.x) this means a numpy
downgrade plus a one-time runtime restart. We dropped `stable-baselines3[extra]`
because it pulls `shimmy[atari]`/`ale-py`, which has no wheel for Colab's Python and
is irrelevant to MuJoCo; we added only tqdm+rich for the progress bar. Ant-v4 is
27-dimensional in gymnasium 0.29 (the brief's 111 is the older contact-force
variant); we use the default 27.

### PPO experts: several re-runs to get usable oracles

- **Walker2d v1 (2M steps): 3010 +/- 901.** It cleared the threshold but the
  variance was high, which would feed inconsistent demonstrations. **Why we re-ran:**
  demonstration quality drives every downstream stage. Adding a **linear LR schedule
  to 0** and extending to 4M tightened it to **4616 +/- 784**.
- **Ant was the hard one.** The Walker2d config (no normalisation) plateaued at
  ~2422. Adding **VecNormalize** + 8M reached 2850.
  - **A mistake we corrected:** an early normalised run was killed at 5M during a
    noisy ~1000-return patch, and we wrongly concluded VecNormalize had failed. The
    full 8M run was actually fine (2850). _Lesson: do not judge a long run from an
    unfinished, high-variance segment._
  - **Why we re-ran again:** the 2850 expert's demonstration dataset **failed the
    quality gate** (only 68% of episodes above two-thirds of the mean; mean episode
    length 763, min return 42). An inconsistent expert cannot be cloned cleanly.
  - **Solved** with the rl-zoo3 Optuna-tuned profile (`tuned_ant`: `n_steps=512`,
    `batch=32`, `lr=1.9e-5`, `gamma=0.98`, VecNormalize, 10M) -> **6293**, gate then
    passed at 90%.

### The 6000+ goal and the env-specific tuning finding

Asked to push experts into the 6000-8000 range, we tried the analogous rl-zoo3
**`tuned_walker` profile on Walker2d - it failed (1640)**, worse than the generic
config and below the 3000 threshold. **Why it failed:** tuned recipes do not
transfer across environments. Walker2d instead reached **6043** with the _standard_
config (`n_steps=2048`, `batch=64`, `lr=3e-4` linear) **plus VecNormalize**, 8M
steps. So Ant wants the low-LR tuned profile and Walker2d wants the standard one;
**VecNormalize was the common decisive lever** on both.

### Behavioural Cloning: a bug, then undertraining, then a device bug

- **Normalisation bug.** A VecNormalize-trained expert consumes normalised
  observations, but the BC students were learning from raw observations with no
  input normalisation and could not fit (Walker2d library BC 1246, scratch 868).
  **Fix:** BC trains and evaluates on the expert's normalised observations.
- **Undertraining.** At a fixed 50 epochs the aggressive Walker2d-6043 student
  reached only ~47% of the expert. The epoch sweep showed it was still improving, so
  **why we re-ran:** the budget was too small. We added validation-loss **early
  stopping** (150-epoch ceiling; spec-recommended) for scratch BC and a 100-epoch
  budget for library BC, which lifted it to ~95%. _The "gap" was mostly training
  budget, not an imitability ceiling._
- **CUDA device bug (Colab).** The imitation-library BC crashed with a device
  mismatch on GPU. **Fix:** force the imitation/SB3 BC paths to CPU (their MLPs are
  tiny and CPU-bound); the from-scratch BC still uses the GPU.

### DAgger: the comparison had to hold training effort fixed

The first DAgger run used only 4 BC epochs per iteration and looked far worse than
BC (Walker2d-6043 best 2402 vs BC 5719). **Why we re-ran:** that is an unfair
comparison against BC's ~100 epochs. The **fair re-run (12 iterations x 25 BC
epochs)** matches or beats BC everywhere and reaches expert level; the Walker2d-6043
curve climbs ~600 -> ~6200 as on-policy aggregation corrects covariate shift.

### Pretraining: warm-start behaviour, and one re-run

Warm-starting PPO from the BC/DAgger policy sharply accelerates learning. We saw a
nuance worth reporting: under Ant's tiny LR the warm-start holds from step 0, while
under Walker2d's larger LR the warm-started actor dips early (the freshly
initialised critic produces noisy advantages) then recovers fast. We also re-ran the
Walker2d pretraining once after a verification smoke-test accidentally overwrote the
real 1.5M artifact, a reminder to point smoke tests at a throwaway output directory.

### Engineering and monitoring lessons

- **`torch.set_num_threads(1)`**: small-MLP PPO on CPU thrashes with BLAS thread
  oversubscription (observed ~9 cores at very low throughput); one thread is far
  faster.
- **Monitor background runs by disk artifacts, not stdout.** SB3's prints are
  block-buffered in a non-TTY background process, so a perfectly healthy run looked
  stalled; this triggered a chain of phantom diagnoses (sleep, then Apple-Silicon
  efficiency-core demotion) before we realised the checkpoint files (landing every
  ~48s, ~2080 steps/s) showed it was fine all along.
- **Compute placement:** PPO is CPU-bound (the MuJoCo sim is not GPU-accelerated),
  so the local Mac is competitive-to-faster than Colab; Colab's value is unattended
  bursts, but its free tier disconnects and `caffeinate` does not stop lid-close
  sleep.
- **Reproducibility:** seeds set everywhere (numpy, torch, python, env, SB3); the
  notebooks re-run the light stages live and are crash-guarded. The discarded
  `tuned1640` Walker2d run was dropped; the second (non-normalised, 4627) Walker2d
  expert is kept for a both-experts comparison.

## 3. Final results

Experts: Walker2d **6043**, Ant **6293** (both above the spec thresholds).

| Metric                  | Walker2d-6043 | Walker2d-4627 | Ant-6293      |
| ----------------------- | ------------- | ------------- | ------------- |
| Library BC              | 5719 (95%)    | 4591          | 6237 (99%)    |
| Scratch BC (early stop) | 4238          | 4160          | 5754          |
| DAgger (fair, 12 iters) | 6208          | 4732          | 6564          |
| Arch winner (5 seeds)   | large 4982    | ~4500 (tight) | ~5900 (tight) |

Stage 5 pretraining, eval return at 1.5M env steps:

|          | PPO scratch | BC + PPO | DAgger + PPO |
| -------- | ----------- | -------- | ------------ |
| Walker2d | 1145        | 5690     | 5451         |
| Ant      | 4965        | 6603     | 6881         |

### Extended requirements (E1, E2; bonus)

Two BC ablations on the from-scratch student, 5 seeds each, both environments
(`noise_sweep.py`, `norm_ablation.py`; figures in notebook 06).

**E1 - noisy expert** (Gaussian noise added to recorded actions). The
environments behave oppositely. Walker2d is highly sensitive: the student is
already below half the expert return at sigma=0.05 (46%) and decays monotonically
to 5% at sigma=0.8, so its collapse point is ~0.05. Ant is robust: it holds at
95-102% of the expert through sigma=0.4 and only bends to 71% at sigma=0.8, never
collapsing in the tested range.

**E2 - observation normalisation** (zero mean / unit variance vs raw). Also
environment-dependent. On Walker2d normalisation is decisive: 4654 (77%) vs 1163
(19%) raw, a 4.0x gap, raw-observation BC effectively fails. On Ant it makes no
difference: 5679 (92%) vs 5946 (96%), within seed noise.

**Theme.** Both reinforce RQ6: Walker2d is fragile to imperfect imitation (needs
clean labels and normalised inputs) while Ant is robust to both, so imitation
difficulty, not state dimensionality, governs sensitivity.

| Bonus                           | Walker2d            | Ant                  |
| ------------------------------- | ------------------- | -------------------- |
| E1 collapse (below half-expert) | sigma >= 0.05       | none up to sigma=0.8 |
| E2 normalised vs raw            | 4654 vs 1163 (4.0x) | 5679 vs 5946 (~1x)   |

### Bonus: SAC (off-policy expert)

To experiment beyond the brief we trained SAC (off-policy, maximum-entropy) and
compared it to the on-policy PPO experts. SAC uses a replay buffer and automatic
entropy tuning and, unlike the PPO setup, **no VecNormalize** (running stats would
corrupt buffered transitions), so it also yields raw-observation demonstrations.
Hyperparameters are the rl-zoo3 MuJoCo recipe (`lr=7.3e-4`, `gamma=0.98`,
`tau=0.02`, `train_freq`/`gradient_steps=8`, gSDE), net `[400,300]` for Ant and
`[256,256]` for HalfCheetah, 3M steps each on CPU (`train_sac.py`).

| Env                        | PPO expert (on-policy) | SAC (off-policy) |
| -------------------------- | ---------------------- | ---------------- |
| Ant-v4                     | 6293 @ 10M steps       | 7295 @ 3M steps  |
| HalfCheetah-v4 (off-brief) | n/a                    | 15387 @ 3M steps |

**Finding:** SAC is much more sample-efficient than PPO here. It beats the Ant PPO
expert (7295 vs 6293) with over 3x fewer environment steps, and on HalfCheetah-v4
it clears the 8000 stretch target by a wide margin. Ant's curve plateaus around
7000-7300 (a 5M extension, stopped at 3.8M, confirmed this: it peaked ~7200 with no
gain over the 3M policy); a sustained 8000 on Ant is above typical SAC ceilings,
whereas HalfCheetah, which has no fall-over failure mode, reaches ~15000.

## 4. Research questions (evidence-backed answers)

**Central question - can imitation reduce PPO's sample complexity?** Yes, decisively.
At 1.5M env steps, PPO from scratch lags (Walker2d ~1.1k, Ant ~5.0k) while
imitation-pretrained PPO is at/near expert level (Walker2d ~5.7k, Ant ~6.6-6.9k).

**RQ1 - How close can BC get to the expert, and what limits the gap?** Close, with
enough training: library BC recovers ~95% (Walker2d-6043) and ~99% (Ant). The
dominant limiter was the training budget (50 epochs gave ~47% on the strong
Walker2d expert; early stopping / more epochs reached ~95%). The residual gap
reflects compounding errors on the expert's near-saturated control (distribution
mismatch). _Evidence: BC table, epoch sweep (nb 03)._

**RQ2 - How does the amount of expert data affect the student?** Performance rises
steeply then saturates: a sharp climb from 5 to ~50 episodes, flat thereafter;
5-10 episodes are low and high-variance across seeds. ~50 demonstrations capture
most achievable performance. _Evidence: dataset-size ablation, 5 seeds (nb 03)._

**RQ3 - Does lower BC loss imply a better policy?** No. Online return and offline
action-MSE do not track each other: at a fixed low validation MSE the evaluation
return still varies by ~1000+ across seeds. Because imitation is evaluated
sequentially, small per-step errors compound, so low supervised loss is necessary
but not sufficient. _Evidence: validation MSE vs return (nb 03)._

**RQ4 - How much does student architecture matter?** Environment-dependent. On
Walker2d a larger MLP (512,512) clearly wins and a tiny (64,64) net underfits; on
Ant all architectures cluster tightly (easy-to-fit mapping, MSE ~5e-4). Capacity
matters most on the harder-to-fit environment. _Evidence: architecture sweep (nb 03)._

**RQ5 - Does DAgger reduce covariate shift vs plain BC?** Yes, clearly. With a fair
budget DAgger matches or beats BC on every config and reaches expert level:
Walker2d-6043 6208 (> BC 5719), Walker2d-4627 4732, Ant 6564 (> BC 6237). The
mechanism is visible in the Walker2d-6043 curve, which climbs ~600 -> ~6200 as
on-policy states are aggregated and the expert labels recovery behaviour.
_Evidence: DAgger iteration curves, BC-vs-DAgger plots (nb 04)._

**RQ6 - Systematic Walker2d vs Ant differences?** Yes. Although Ant is
higher-dimensional and 3D, it is _easier to imitate_ (BC 99% vs 95%, action map fits
to MSE ~5e-4, architecture barely matters, warm-start stable), whereas Walker2d is
harder (aggressive near-saturated control compounds errors, needs more training and
a larger net). Notably, expert _training_ difficulty was the opposite (Ant needed
the tuned config), so training difficulty and imitation difficulty are distinct
axes. _Evidence: cross-environment results (nb 05)._
