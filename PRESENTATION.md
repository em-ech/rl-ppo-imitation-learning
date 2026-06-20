# Presentation Draft & Guidelines
### Training an Agent with PPO and Imitation Learning — Walker2d / Ant

This document is the **speaking script and slide plan** for the 15-minute class presentation
(+ 5 min Q&A). It is the source for `PRESENTATION.pptx`. Every number here traces back to our
experiments (see `info/*.md`, `PROJECT_OVERVIEW.md`).

---

## PART A — Presenting Guidelines

### The golden rule (from the brief, §11)
> *"This is a **research talk, not a tutorial**. Lead with a clear research question, present
> your methodology **briefly**, then spend the **majority of the time on results and
> discussion**."*

Because the assignment brief is **identical for every group**, the methodology is **not** where
we win points. We compress it to ~2 slides and spend our time on **results, the six research
questions, and the engineering choices + surprises that are unique to us.**

### Time budget (~15 min, slide count is flexible)
| Section | Slides | Time | Owner |
| :-- | :-- | :-- | :-- |
| 1. Title + Goal | 1 | 1.0 min | **Member 1** |
| 2. Research question + headline finding | 1–2 | 2.0 min | **Member 1** |
| 3. Pipeline at a glance (how we built it) | 2 | 2.0 min | **Member 2** |
| 4a. Main results table + central thesis | 2 | 2.5 min | **Member 3** |
| 4b. RQ1–RQ3 | 2–3 | 2.5 min | **Member 3 → 4** |
| 4c. RQ4–RQ6 | 2–3 | 2.5 min | **Member 4** |
| 5. Deep dive: our implementation + surprises | 3–4 | 2.0 min | **Member 5** |
| 6. Conclusion + future work | 1 | 0.5 min | **Member 5** |
| **Q&A** | backup | 5.0 min | **All** |

> Slide count is a guide, not a rule — add or merge slides as long as everything is covered
> clearly. Target ~15–18 content slides + backup.

### Delivery tips
- **Open with the punchline.** Say the answer to the central question in the first 90 seconds,
  then spend the talk proving it.
- **One key number per slide.** Don't crowd; let the headline number land.
- **Play the videos live.** The side-by-side expert-vs-student clips (`videos/`) are the most
  persuasive 10 seconds in the talk — use them on the BC / DAgger slides.
- **Lean into "what surprised us."** The brief explicitly rewards this. Our best material:
  Ant is *easier to imitate than Walker despite more dimensions*, and the *warm-start dip*.
- **Every number is from an experiment.** Never mix opinion with results.
- **All 5 members speak.** Section ownership is in the table above; rehearse hand-offs.

### Asset notes (read before building slides)
- **Videos exist** in `videos/`: `expert_vs_student_Walker2d-v4.mp4`,
  `expert_vs_student_Ant-v4.mp4`, `comparison_*.mp4`. These satisfy deliverable M6.
- **Figures are NOT committed.** The `outputs/*.png` charts must be **regenerated** by running
  the notebooks `notebooks/0*.ipynb` (or the scripts `bc_experiments.py`, `arch_sweep.py`,
  `noise_sweep.py`, `norm_ablation.py`, `pretraining.py`) before they can be embedded. Slides
  below reference each chart by its **expected filename** so you can drop them in.

---

## PART B — Slide-by-Slide Script

> Each slide has: **Visual** (what's on screen), **Bullets** (what the audience reads), and
> **Script** (what the presenter says — rehearse close to verbatim).

---

### SLIDE 1 — Title
**Visual:** Project title; environment thumbnails (Walker2d + Ant); all 5 names.
**Bullets:**
- Training an Agent with PPO and Imitation Learning — Walker2d / Ant
- Reinforcement Learning Group Project · Dr. Jaume Manero
- [Member 1] · [Member 2] · [Member 3] · [Member 4] · [Member 5]

**Script (Member 1):**
> "Good morning. Our project is about a two-stage pipeline: first we train an expert with
> Reinforcement Learning — PPO — and then we use Imitation Learning to copy that expert into a
> new agent. We ran the full pipeline on both MuJoCo locomotion environments, the 2D Walker
> and the 3D Ant. I'm [Member 1], and my teammates [names] will each take a section."

---

### SLIDE 2 — The question and the answer (HEADLINE)
**Visual:** The central research question big at top; below it, the punchline chart
`outputs/pretraining_Walker2d.png` (or a 2-bar "scratch vs warm-start @1.5M" comparison).
**Bullets:**
- **Central question: Can imitation learning reduce the *sample complexity* of PPO?**
- **Answer: Yes — decisively.**
- At 1.5M steps: PPO-from-scratch is still falling over; imitation-warm-started PPO is already
  at expert level.

**Script (Member 1):**
> "Here is the one question the whole project is built around: *can imitation learning reduce
> the sample complexity of PPO?* — meaning, can we reach expert performance with far fewer
> environment interactions? Our answer is yes, decisively. This is the headline: after only
> 1.5 million steps — a tiny budget for MuJoCo — a PPO agent trained from scratch is still
> flailing, around 1,100 on Walker and 5,000 on Ant. But if we warm-start that same PPO with
> an imitation-learned policy, it's already at expert level — about 5,700 on Walker and 6,600
> on Ant. The rest of the talk is us proving how we got here."

---

### SLIDE 3 — The pipeline in one picture
**Visual:** 5-box flow diagram (reuse the brief's Figure 3 concept):
PPO Expert → Demonstrations → Behavioural Cloning → DAgger → Imitation-as-PPO-Pretraining.
**Bullets:**
- ① PPO Expert (the teacher) → ② Demos (the textbook) → ③ BC (the student)
- ④ DAgger (interactive correction) → ⑤ Warm-start PPO (the payoff)
- *Same brief for every group — so we'll focus on **how we implemented it** and what we found.*

**Script (Member 2):**
> "I'm [Member 2]. Quickly, the pipeline. We train a PPO **expert** — the teacher. We roll it
> out to record a dataset of demonstrations — the textbook. We train a **student** by
> Behavioural Cloning, pure supervised learning. We then fix the student's weaknesses with
> **DAgger**, and finally we use the student to **warm-start** a fresh PPO run. Since every
> group has this exact brief, we'll keep the *how-it-works* short and spend our time on results
> and on the implementation decisions that are specifically ours."

---

### SLIDE 4 — How we implemented each phase (methodology, compressed)
**Visual:** Compact 5-row table: phase | what we did | key choice.
**Bullets:**
- **Expert:** PPO via Stable-Baselines3; Walker 8M steps → 6043, Ant 10M steps → 6293
- **Demos:** 100 deterministic episodes; strict 90% quality gate; raw obs saved for ablations
- **BC:** library (`imitation`) **and** from-scratch PyTorch; MSE on (state→action)
- **DAgger:** 12 iters × 5k steps; expert relabels the student's own states
- **Pretraining:** inject student weights into PPO, fine-tune 1.5M steps vs scratch

**Script (Member 2):**
> "One slide on the methods. The expert is PPO from Stable-Baselines3 — eight million steps for
> Walker, ten million for Ant. We collected 100 deterministic demonstration episodes and gated
> them on quality. For Behavioural Cloning we did it twice — once with the `imitation` library
> and once from scratch in PyTorch — both just minimizing the squared error between the
> student's action and the expert's. DAgger runs twelve interactive rounds where the expert
> relabels the states the student actually visits. And pretraining injects that student into a
> new PPO and fine-tunes. Now [Member 3] will take the results."

---

### SLIDE 5 — Main results: the comparison table
**Visual:** The money table.
| Metric | Walker2d | Ant |
| :-- | :-- | :-- |
| PPO Expert | 6043 | 6293 |
| Behavioural Cloning | 5719 (95%) | 6237 (99%) |
| DAgger | 6208 (102%) | 6564 (104%) |
| Warm-start PPO @1.5M | 5690 | 6603 |
| PPO from scratch @1.5M | 1145 | 4965 |

**Bullets:**
- BC alone recovers **95% (Walker) / 99% (Ant)** of the expert
- DAgger **matches or beats** the expert (102% / 104%)
- Warm-start reaches expert level at a fraction of the budget

**Script (Member 3):**
> "I'm [Member 3]. This is the single most important table in the talk. Read it top to bottom.
> Our PPO experts score about 6,000 on both robots. Pure Behavioural Cloning already recovers
> 95% of that on Walker and an astonishing 99% on Ant. DAgger pushes past the expert — 102 and
> 104%. And the bottom two rows are the payoff: warm-started PPO reaches expert level at 1.5
> million steps, while from-scratch PPO at the same budget is nowhere close. Every number here
> is a measured evaluation return, not a training loss."

---

### SLIDE 6 — Central thesis: sample efficiency curve
**Visual:** Return-vs-timesteps learning curves: scratch vs BC+PPO vs DAgger+PPO, both envs.
Charts: `outputs/pretraining_Walker2d.png`, `outputs/pretraining_Ant.png`.
**Bullets:**
- The gap *is* the saved compute — imitation skips the random-exploration phase
- Warm-started curves start high and stay high; scratch curve crawls
- This is the direct answer to the central question

**Script (Member 3):**
> "Here's that result as a learning curve — return on the y-axis, environment steps on the x.
> The scratch agent, in blue, spends its whole budget just learning not to fall over. The
> warm-started agents start near expert level from step zero and stay there. The whole shaded
> gap between the curves is compute we *didn't* have to spend. That gap is the answer to our
> central question: imitation learning buys you sample efficiency by skipping random
> exploration entirely."

---

### SLIDE 7 — RQ1 & RQ2: How good is BC, and how much data does it need?
**Visual:** Left: `outputs/bc_epoch_sweep_*.png`. Right: `outputs/bc_ablation_data_size_*.png`.
**Bullets:**
- **RQ1 — Gap to expert:** 95% (Walker) / 99% (Ant); residual gap = covariate shift
- **RQ2 — Data:** steep gains 5→50 episodes, then **saturates**; 100 ≈ 50
- 5–10 episodes = poor and high-variance across seeds

**Script (Member 3 → hand to Member 4):**
> "Research question one: how close can pure imitation get, and what limits it? Very close —
> 95 and 99% — and the limiter is *covariate shift*, the compounding-error problem we'll come
> back to. Question two: how much expert data do you actually need? Performance climbs steeply
> from 5 to 50 episodes and then flatlines — 100 episodes is barely better than 50. With only
> 5 or 10 episodes the student is poor and wildly seed-dependent. So there's a clear
> 'enough-is-enough' point. [Member 4] will take the next questions."

---

### SLIDE 8 — RQ3: Does lower loss mean a better policy?
**Visual:** Validation-MSE vs evaluation-return scatter across seeds (no clean correlation).
**Bullets:**
- **No.** Offline MSE measures static accuracy; online return measures *walking*
- One tiny error → unseen posture → fall. Sequential ≠ i.i.d.
- A "perfect-MSE" model can still score poorly

**Script (Member 4):**
> "I'm [Member 4]. Question three is a trap a lot of people fall into: surely a lower training
> loss means a better agent? No. The MSE measures whether you predict the right action on a
> *frozen* snapshot. But the robot walks *sequentially* — one small mistake puts it in a
> posture it never saw in training, and it falls. So a model with a beautiful low MSE can make
> one fatal error and score badly, while a slightly worse-MSE model walks fine. Offline metrics
> and online performance can genuinely diverge."

---

### SLIDE 9 — RQ4: How much does student architecture matter?
**Visual:** `outputs/bc_arch_sweep_Walker2d.png` next to `outputs/bc_arch_sweep_Ant.png`.
**Bullets:**
- **It depends entirely on the environment**
- **Walker:** needs a large net `[512,512]`; `[64,64]` underfits badly
- **Ant:** all sizes cluster at the same high performance (MSE ~5e-4)

**Script (Member 4):**
> "Question four: does the student's network architecture matter? The honest answer is: it
> depends on the robot. On Walker, it matters enormously — a tiny 64-by-64 network underfits
> and the agent can't walk; you need a big 512-by-512 network to capture its control dynamics.
> On Ant, architecture barely matters at all — every network we tried, small or large, lands at
> basically the same performance. That contrast is a preview of our biggest finding."

---

### SLIDE 10 — RQ5: Does DAgger beat plain BC?
**Visual:** `outputs/bc_vs_dagger_Walker2d.png` — DAgger iteration curve climbing to ~6200.
**Bullets:**
- **Yes, clearly** — DAgger matches/beats BC on every config (102% / 104%)
- You can watch it work: Walker climbs ~600 → ~6200 iteration by iteration
- Mechanism: expert labels recovery from the student's *own* mistakes (O(εT) not O(εT²))

**Script (Member 4):**
> "Question five: does DAgger actually fix covariate shift in practice, or is it just nice
> theory? Clearly yes. Given a fair compute budget, DAgger matches or beats plain BC on every
> single configuration, ending slightly *above* the expert. And you can literally watch the
> mechanism work on Walker — the curve climbs from about 600 to 6,200 iteration by iteration as
> the expert teaches the student to recover from the exact mistakes it makes. Theory says
> DAgger turns quadratic error growth into linear; our curves show it."

---

### SLIDE 11 — RQ6: Walker vs Ant — the surprise
**Visual:** Two columns contrasting the robots; the noise & normalization ablation charts
(`outputs/bc_noise_sweep_*.png`, `outputs/bc_norm_ablation_*.png`).
**Bullets:**
- **Ant is harder to *train* but easier to *imitate*; Walker is the reverse**
- **Imitation difficulty ≠ state dimensionality** (Ant has more dims, yet is robust)
- Walker collapses at tiny noise (σ=0.05) and demands normalization; Ant shrugs both off

**Script (Member 4 → hand to Member 5):**
> "Question six ties it all together and it genuinely surprised us. Everyone assumes Ant is
> harder to imitate — it's 3D, more joints, far more observation dimensions. The opposite is
> true. Ant is incredibly robust: it imitates well with noisy labels, with no normalization,
> and with tiny networks. Walker is fragile — inject even 5% action noise and it collapses;
> remove normalization and it fails completely. The lesson: *imitation difficulty is
> disconnected from dimensionality* — it's about how saturated and unforgiving the control is.
> [Member 5] will take the deep dive on how we engineered around this."

---

### SLIDE 12 — Deep dive: the engineering that made the expert work
**Visual:** Before/after numbers; small code snippet of `linear_schedule`.
**Bullets:**
- **Linear LR decay** tightened Walker from 3010±901 → 4616±784 → ~6043
- **Ant needed a different profile** (`tuned_ant`: lr 1.9e-5, clip 0.1) — defaults plateaued at 2422
- **VecNormalize** was decisive; **strict 90% quality gate** caught "lucky" experts
- Robustness: thread pinning, checkpoint **+ VecNormalize** resume

**Script (Member 5):**
> "I'm [Member 5]. Since the brief is shared, here's what's actually *ours*. Getting the expert
> stable took real work. A constant learning rate gave us a Walker that swung wildly —
> 3,000 plus-or-minus 900. Switching to a linear learning-rate decay, which settles the policy
> at the end of training, tightened that dramatically and got us to 6,000. Ant was worse: the
> default config plateaued at 2,400 and never recovered until we switched to an Optuna-tuned
> profile with a much smaller learning rate and tighter clipping. Normalization was decisive,
> and a strict quality gate — 90% of episodes must clear two-thirds of the mean — caught
> experts that were just getting lucky."

---

### SLIDE 13 — Deep dive: rigor and fixes
**Visual:** Icons for: 5-seed error bars; early stopping; "fixed broken Listing 10".
**Bullets:**
- **5-seed ablations** everywhere — error bars, not lucky single runs
- **Validation-loss early stopping** in our from-scratch BC
- **We fixed the brief's DAgger code** (Listing 10 crashed on `imitation` v1.0.0)
- Device management: library BC on CPU, from-scratch BC on GPU/MPS

**Script (Member 5):**
> "On rigor: we didn't trust single runs — every ablation is five seeds with error bars, the
> standard in RL research. Our from-scratch BC uses validation-loss early stopping so we keep
> the best epoch, not the last. And a fun one: the DAgger sample code in the assignment PDF
> actually crashes on the current version of the `imitation` library — it passes `None` where a
> trainer is required. We diagnosed the API change and rewrote it so it runs."

---

### SLIDE 14 — Deep dive: the surprises worth discussing
**Visual:** The warm-start "dip" curve; SAC vs PPO bar (Ant 7295 vs 6293).
**Bullets:**
- **Warm-start dip:** IL trains the *actor* only; the *critic* starts random → a brief
  performance crash before recovery
- **E1/E2 ablations** confirm the Walker-fragile / Ant-robust theme
- **Bonus SAC:** off-policy beat PPO's sample efficiency — Ant 7295 in 3M (vs 6293 in 10M);
  HalfCheetah 15387

**Script (Member 5):**
> "Three things that surprised us. First, when we warm-start Walker, performance *dips* hard
> before recovering — because imitation only trains the actor, the part that chooses actions.
> The critic, which judges how good a state is, starts random and feeds bad gradients until it
> catches up. Second, our noise and normalization ablations confirmed the fragility story.
> And as a bonus, we tried off-policy SAC: it crushed PPO on sample efficiency — 7,295 on Ant
> in just 3 million steps versus PPO's 6,293 in ten — but we kept PPO for the pipeline because
> the brief required it and because SAC can't use the normalization our pipeline depends on."

---

### SLIDE 15 — Conclusion & future work
**Visual:** Three takeaways; one-line future-work list.
**Bullets:**
- **Imitation slashes PPO sample complexity** — expert-level at 1.5M vs millions from scratch
- **DAgger > BC**; **offline loss ≠ online skill**; **difficulty ≠ dimensionality**
- **Next:** harder tasks (targets, perturbations), critic warm-start to kill the dip,
  transformer/compressed students

**Script (Member 5):**
> "To conclude: imitation learning dramatically cuts PPO's sample complexity — we reach
> expert level in a fraction of the steps. DAgger beats plain cloning, a low training loss does
> not guarantee a good policy, and imitation difficulty has nothing to do with how many
> dimensions the robot has. With two more weeks we'd warm-start the critic too — to kill that
> dip — push to harder tasks with targets and perturbations, and try compressing the expert
> into a smaller or transformer student. Thank you — we're happy to take questions."

---

## PART C — Q&A Prep & Backup Slides

### Anticipated questions (with crisp answers)
- **Q: If SAC is so much more sample-efficient, why use PPO?**
  The brief requires PPO (on-policy) to study imitation. Also, SAC's replay buffer is
  incompatible with `VecNormalize` (changing running stats corrupts buffered transitions), so
  SAC must use raw observations — which makes the BC stage much more fragile. PPO let us keep
  normalized observations consistent across the whole pipeline.
- **Q: Why does DAgger only barely beat BC on Ant?**
  Ant is already easy to imitate — BC alone gets 99%, so there's almost no covariate-shift gap
  left for DAgger to close. The win is much larger on the fragile Walker.
- **Q: Why can a model with lower MSE be a worse walker?**
  MSE is i.i.d. and static; control is sequential. One small error moves the agent into an
  unseen state where errors compound. So offline accuracy and online return diverge (RQ3).
- **Q: Why did Ant need a different hyperparameter profile?**
  Ant is a 3D robot with unstable contact dynamics; the default PPO config plateaued at ~2422.
  The Optuna-tuned `tuned_ant` profile (lower lr 1.9e-5, tighter clip 0.1) prevents
  catastrophic updates that flip the robot.
- **Q: What is the warm-start dip?**
  IL trains only the actor; the critic is random at step 0, so early advantage estimates are
  noisy and temporarily wreck the policy until the critic catches up.

### Backup slides to have ready
- Full PPO hyperparameter tables (standard vs `tuned_ant`).
- Dataset EDA (`outputs/dataset_analysis_*.png`): return distribution + per-joint actions.
- Full E1 noise sweep and E2 normalization numbers (Walker 4654 vs 1163 raw = 4× gap).
- Library-BC vs from-scratch-BC comparison (M4).
- TensorBoard expert training curves (M1).

---

## Build checklist
- [ ] Fill in the 5 real team-member names (Slide 1 + title slide).
- [ ] Regenerate `outputs/*.png` from the notebooks, then embed on Slides 2, 6, 7, 9, 10, 11, 14.
- [ ] Embed/queue `videos/expert_vs_student_*.mp4` on the BC/DAgger slides.
- [ ] Rehearse hand-offs between members; confirm total ≤ 15 min.
