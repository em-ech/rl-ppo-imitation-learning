# Group Project Plan: PPO + Imitation Learning (Walker2d and Ant)

Source spec: `RL_v2_MBD_GroupPractice-2.pdf` (Dr. Jaume Manero).
Environments in scope: `Walker2d-v4` and `Ant-v4` (both required).
Compute: local CPU for development and validation, Google Colab GPU for heavy sweeps.
Central question: can imitation learning reduce the sample complexity of PPO in continuous control locomotion tasks?

---

## 1. Decisions and conventions

These hold across every notebook and experiment.

| Convention      | Value                                     | Reason                                                  |
| --------------- | ----------------------------------------- | ------------------------------------------------------- |
| Python          | 3.10 or 3.11 venv                         | torch / SB3 / imitation lack reliable 3.13+ wheels      |
| Seeds           | {0, 1, 2, 3, 4}                           | 5 seeds for any reported mean and std                   |
| Eval protocol   | 20 episodes, `deterministic=True`         | matches the spec's expert evaluation                    |
| Expert net arch | `pi=[256,256]`, `vf=[256,256]`, Tanh      | matches PPO expert so BC student capacity is comparable |
| Return targets  | Walker2d > 3000, Ant > 4000               | spec thresholds, the gate to Phase 2                    |
| net_arch syntax | `dict(pi=[256,256], vf=[256,256])`        | SB3 v2.x removed the deprecated outer-list form         |
| Saved expert    | `best_model` from EvalCallback, not final | spec instruction; final may be overfit                  |
| Demo actions    | deterministic mean                        | less noisy BC targets                                   |

Everything random is seeded: numpy, torch, the env, and SB3. Notebooks must run top to bottom with no manual intervention (spec penalises crashes on fresh run).

---

## 2. Fixes required to the professor's provided code

The listings are a skeleton, not runnable as-is. These must be addressed before experiments start.

| #   | Listing  | Problem                                                                                           | Fix                                                                                     |
| --- | -------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| F1  | all      | Python 3.13/3.14 only on machine                                                                  | create a 3.10/3.11 venv                                                                 |
| F2  | 2, 6, 10 | `net_arch=[dict(...)]` deprecated                                                                 | use `net_arch=dict(pi=..., vf=...)`                                                     |
| F3  | 10       | DAgger: `bc_trainer=None`, student model never passed, stale API (`round_num`, `bc_train_kwargs`) | rewrite against installed `imitation` version, pass a real `BC` trainer                 |
| F4  | none     | Stage 5 (BC+PPO fine-tune) has no code                                                            | build `bc_bridge.py` to copy BC weights into an SB3 PPO policy                          |
| F5  | 4, 9     | per-episode arrays and `episode_lengths.npy` never saved, but ablation and EDA need them          | save per-episode obs/acts plus an `episode_starts` index                                |
| F6  | 6, 9     | `rng` and `EXPERT_MEAN` undefined                                                                 | define `rng = np.random.default_rng(SEED)`, load expert mean from `episode_returns.npy` |
| F7  | 6        | `Transitions` crosses episode boundaries, `dones` all False                                       | harmless for pure BC (ignores next_obs/dones); add a comment so graders do not flag     |
| F8  | 6 vs 7   | imitation BC 50 epochs vs scratch BC 100                                                          | hold epochs/lr/batch/seed equal for the M4 comparison                                   |

---

## 3. Repository structure

```
GroupProject/
  notebooks/
    01_ppo_expert.ipynb        # M1
    02_data_collection.ipynb   # M2
    03_bc_student.ipynb        # M3, M4, M5, M8
    04_dagger.ipynb            # M7
    05_pretraining.ipynb       # Stage 5 sample-efficiency study
  src/
    envs.py                    # env factory, seeding, VecNormalize wrapper (E2)
    collect.py                 # rollout + per-episode saving (fixes F5)
    bc_scratch.py              # from-scratch StudentPolicy + train loop
    bc_bridge.py               # copy BC weights into SB3 PPO policy (F4)
    eval.py                    # shared evaluate + video rendering (M6)
    plotting.py                # shared figure helpers (DRY)
    seeding.py                 # one seed function for numpy/torch/env/SB3
  configs/                     # one file per experiment config
  models/  data/demonstrations/  outputs/  videos/
  requirements.txt   README.md
```

Shared logic (eval, plotting, env build, seeding) lives in `src/` so the notebooks do not duplicate it.

---

## 4. Experiment matrix

Each stage lists the strategies compared, the metric, and the deliverable it satisfies.

### Stage 1: PPO expert (M1, RQ1)

Goal: a competent expert, treated as a means, not the research focus.

- Strategy pilots (short runs, pick one config): gamma {0.99, 0.995}, gae_lambda {0.9, 0.95}, lr {3e-4 constant, 3e-4 linear-to-0}, n_steps {2048, 4096}, ent_coef {0.0, 0.01}.
- Decision rule: 2 to 3 pilot runs per env at 0.5M steps, lock the best, then a full 2M (Walker2d) / 3M+ (Ant) run.
- Outputs: TensorBoard return curves, best checkpoint, expert video.

### Stage 2: demonstration dataset (M2)

- Collect 100 episodes per env (>= 50 required), deterministic actions.
- Save observations, actions, episode_returns, episode_starts (fixes F5).
- EDA: return distribution, per-joint action distribution, trajectory-length histogram.
- Quality gate: >= 90% of episodes above two thirds of the eval mean.

### Stage 3: Behavioural Cloning (M3, M4, M5, M8, RQ1, RQ3, RQ4)

The core comparison stage.
| Experiment | Axis | Metric | Deliverable |
|---|---|---|---|
| Library vs scratch BC | implementation | env return + MSE, equal hyperparams | M3, M4 |
| Dataset-size ablation | {5,10,20,50,100} eps x 5 seeds | mean return +/- std vs data | M5, RQ2 |
| Architecture sweep | small MLP, default, large MLP, skip-connections | gap to expert | M8, RQ4 |
| Offline vs online | val MSE vs env return | divergence | RQ3 |

### Stage 4: DAgger (M7, RQ5)

- Rewrite Listing 10 against the installed imitation API.
- Compare BC vs DAgger as a function of iterations; plot aggregated dataset growth.
- Metric: return, stability, gap to expert narrowing with on-policy data.

### Stage 5: imitation as PPO pretraining (sample efficiency, central question)

Five strategies on return vs environment timesteps:

1. PPO from scratch
2. BC only
3. DAgger only
4. BC + PPO fine-tune
5. DAgger + PPO fine-tune

Headline metric: timesteps to reach a target return (sample efficiency), not only final return.

### Bonus

- E1: noisy expert. Add Gaussian noise to recorded actions at increasing levels; find where BC collapses.
- E2: observation normalization on vs off; effect on convergence and final return.

---

## 5. Runtime budget (rough)

| Task                         | Walker2d      | Ant             | Where                     |
| ---------------------------- | ------------- | --------------- | ------------------------- |
| PPO expert pilots            | ~10 min each  | ~20 min each    | local                     |
| PPO expert full              | ~30 min (2M)  | ~60-90 min (3M) | local or Colab            |
| Demo collection              | ~5 min        | ~10 min         | local                     |
| BC single run                | a few min     | a few min       | local (GPU helps scratch) |
| Ablation 5 sizes x 5 seeds   | ~1-2 h        | ~1-2 h          | Colab                     |
| DAgger 10 iters              | ~30-60 min    | ~1 h+           | Colab                     |
| Pretraining grid (5 methods) | several hours | several hours   | Colab                     |

Strategy: develop and smoke-test every notebook locally on Walker2d, then run the seed-heavy sweeps on Colab, and run Ant after the Walker2d pipeline is validated end to end.

---

## 6. Deliverables checklist (from Section 10)

- [ ] `notebooks/01_ppo_expert.ipynb` with TensorBoard screenshots and hyperparameter discussion
- [ ] `notebooks/02_data_collection.ipynb` with dataset stats and the two EDA plots
- [ ] `notebooks/03_bc_student.ipynb` with loss curves (library + scratch), eval, ablation plot
- [ ] `notebooks/04_dagger.ipynb` with BC vs DAgger plot and written discussion
- [ ] `models/` with `best_model.zip` and `bc_student.pt`
- [ ] `videos/expert_vs_student.mp4` (M6)
- [ ] `requirements.txt` with pinned versions
- [ ] `presentation.pdf` (15 min research talk)
- [ ] `README.md` with reproduction steps
- [ ] group member names on every notebook and the title slide

Research questions RQ1 to RQ6 drive the final discussion and must be answered with evidence (curves, ablations, DAgger plots, videos), not qualitatively.

---

## 7. Suggested group task split

Four work tracks that can run in parallel once Stage 1 and 2 produce the expert and dataset:

1. PPO expert + pretraining (Stages 1 and 5)
2. BC library + ablation (Stage 3, M3/M4/M5)
3. BC scratch + architecture/bonus (M4/M8/E1/E2)
4. DAgger (Stage 4) + video + presentation assembly

Stages 1 and 2 are the critical path; everything else depends on the expert checkpoint and the demonstration dataset.

---

## 8. Immediate next actions

1. Create the Python 3.10/3.11 venv and `requirements.txt`; run the MuJoCo sanity check (Listing 1).
2. Scaffold `src/` modules and the five notebook stubs.
3. Run a Walker2d PPO pilot to validate the full pipeline before scaling to Ant.
