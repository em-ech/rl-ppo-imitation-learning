# Phase 1 Deep Dive: PPO Expert Training

This phase is the foundation of the entire project because the quality of the "expert" agent dictates the quality of the demonstrations. If the expert is bad, the student will imitate bad behavior.

Here is how the team trained the expert, the key methodological points, and what makes your implementation stand out from a generic run-of-the-mill tutorial. All of these insights are derived from the team's `PROJECT_OVERVIEW.md` log and the source code in `src/ppo.py`.

## 1. How It Is Trained
The team used `stable-baselines3` (SB3) to train the experts using **Proximal Policy Optimization (PPO)**. 
The training script (`train_expert.py`) sets up multiple parallel physics simulations (vectorized environments) to collect data faster. *(Note: While using `make_vec_env` to parallelize simulations is the standard and recommended way to train PPO in SB3—and is explicitly requested in the project guidelines—it's worth noting as a core part of the architecture).*

* **Walker2d** was trained for **8 Million steps**, reaching a score of ~6043.
* **Ant** was trained for **10 Million steps**, reaching a score of ~6293.

## 2. Key Methodological Points

The team hit several roadblocks that required smart methodological pivots. These were documented meticulously in the `PROJECT_OVERVIEW.md`:

* **Observation Normalization is Crucial:** The team wrapped the environments in `VecNormalize`. This scales the raw inputs (like joint angles and velocities) to have a mean of 0 and a variance of 1. As noted in the `PROJECT_OVERVIEW.md`, when the team tried to train the *Ant* environment without normalization (using the standard Walker2d config), it plateaued and failed at a score of **~2422**. Once `VecNormalize` was added, the performance skyrocketed. *(Note: Investigating observation normalization was explicitly requested in the project guidelines under the Extended Requirements (E2), and the team's empirical results strongly validated its inclusion).*
* **The "Linear Learning Rate" Schedule:** Standard PPO with a constant learning rate produced an expert with very high variance. According to the `PROJECT_OVERVIEW.md`, an early Walker2d expert scored **3010 +/- 901** (meaning performance fluctuated wildly between ~2000 and ~4000 depending on the episode). 
  * **How they fixed it:** In `src/ppo.py`, they wrote a custom Python function `linear_schedule(initial: float)`. SB3 allows you to pass a function instead of a constant for the `learning_rate` parameter. This function takes the remaining training progress (from 100% down to 0%) and multiplies it by the initial rate (`3e-4`). This means the learning rate slowly drops to exactly `0` by the end of training. This forces the agent to make smaller and smaller updates, "settling" into a very stable policy. *(Conceptually, this is indeed very similar to how an epsilon-greedy policy decays epsilon over time to transition from "exploration" to "exploitation". While epsilon decay reduces random action selection, a decaying learning rate reduces the step size the neural network takes during updates—both help the agent stabilize its final behavior).* This fix tightened the variance to **4616 +/- 784** on the next run, and eventually helped reach **~6043**.
  * **Why Linear over other types of decay?**
    * **Exponential / Geometric Decay:** In exponential decay, the learning rate drops off very quickly at the beginning and trails off slowly. For PPO (an on-policy algorithm that constantly needs to adapt to newly collected states), dropping the learning rate too fast early on can cause the agent to get stuck in a suboptimal walking pattern before it has fully explored the environment.
    * **Step Decay:** Step decay drops the learning rate by massive chunks at specific milestones (e.g., dropping by 50% every 2M steps). In continuous control physics tasks, a sudden cliff in the learning rate can disrupt the delicate trust-region math of PPO, sometimes causing sudden crashes in performance.
    * **Linear Decay (The Winner):** A linear decay is the most robust and standard approach for PPO. It keeps the learning rate reasonably high for the vast majority of training so the agent can continually adapt to its own expanding capabilities, only gently guiding it to zero at the very end.
* **Environment-Specific Tuning:** One of the most important findings was that **hyperparameters do not transfer across environments**. 
  * For **Walker2d**, the standard SB3 configuration (`n_steps=2048, batch_size=64, learning_rate=3e-4` with the linear decay) worked perfectly.
  * For **Ant**, this standard configuration completely failed. As noted in the overview, the team had to switch to an aggressively tuned profile (`tuned_ant` from `rl-zoo3`). 
  * **The differences:** The `tuned_ant` profile (visible in `src/ppo.py`) uses a much smaller, constant learning rate (`1.9e-5`), smaller batch sizes (`32` instead of `64`), smaller rollouts (`n_steps=512`), and tighter clipping ranges (`0.1`). Ant is a 3D robot, making its physics much more unstable than the 2D Walker. The lower learning rate and tighter clipping prevent catastrophic neural network updates that would cause the Ant to "forget" how to walk and flip over.
* **Strict Demonstration Quality Gates:** The team didn't just look at the final average score. They realized that an expert averaging 2850 could actually be failing 30% of the time and getting lucky 70% of the time. They implemented a strict quality gate to ensure the demonstrations were clean: *At least 90% of the episodes must achieve returns above two-thirds of the evaluation mean.* *(Note: This was implemented to satisfy a strict requirement from page 12 of the project guidelines PDF, designed to catch "lucky" but inconsistent experts before moving to Phase 2).*

## 3. What Makes This Implementation Stand Out? (The Differentiators)

If you compare this repository to another group following the exact same PDF guideline, your team implemented several advanced engineering details that make this codebase significantly more robust:

> [!TIP]
> **CPU Thread Pinning (`torch.set_num_threads(1)`)**
> *The Problem:* PPO using tiny MLPs (like `[256, 256]`) running on a CPU can actually trigger "BLAS oversubscription". PyTorch tries to use all 8 or 10 cores of the CPU for a tiny matrix multiplication, causing the threads to fight each other and dropping the speed to <40 steps per second.
> *The Fix:* The team explicitly pinned PyTorch to 1 thread at the top of the script. This forces the small matrix multiplications to happen on a single core, which is drastically faster for this specific architecture.

> [!NOTE]
> **Robust Checkpoint Resuming**
> Training for 10M steps takes hours. If a laptop goes to sleep or Google Colab disconnects, you lose everything. 
> The team built a robust resuming mechanism in `train_expert.py`. It automatically scans the `models/` directory for the latest checkpoint `ppo_checkpoint_X_steps.zip` and seamlessly resumes training exactly where it left off. More impressively, **it also loads and resumes the `VecNormalize` running statistics**, ensuring the normalization math isn't corrupted by a restart.

In summary, the team didn't just call `model.learn()`. They engineered a robust, crash-proof pipeline that adapts hyperparameters to the specific physics of the robot and guarantees high-quality, normalized data for the imitation phases downstream.

## 4. Final Results of Phase 1

After applying all the methodological fixes, the final PPO experts successfully cleared the minimum thresholds and provided a stable, high-quality foundation for the imitation learning phases.

| Metric | Walker2d | Ant |
| :--- | :--- | :--- |
| **Final Score** | **~6043** | **~6293** |
| **Minimum Threshold** | > 3000 | > 4000 |
| **Training Steps** | 8 Million | 10 Million |
| **Hyperparameter Profile** | Standard + Linear LR | Optuna Tuned (`tuned_ant`) |

### What do these profiles mean?
* **Standard + Linear LR:** This means the team used the default, out-of-the-box parameters provided by the `stable-baselines3` library for PPO (e.g., `batch_size=64`, `n_steps=2048`), but swapped the constant learning rate for the custom linear decay schedule discussed in Section 2.
* **Optuna Tuned:** `Optuna` is a popular open-source hyperparameter optimization framework. The developers of `stable-baselines3` (in their `rl-zoo3` repository) ran massive, automated searches using Optuna to find the mathematically "perfect" PPO parameters for specific environments. The team grabbed these specific tuned values for Ant because the default ones failed.

### *(Bonus)* The SAC Experiment (Off-Policy vs On-Policy)
The team also went beyond the requirements and experimented with training an **off-policy agent (SAC - Soft Actor-Critic)** to compare against the on-policy PPO experts. The SAC algorithm proved to be significantly more sample-efficient on the Ant environment, reaching a score of **7295 in just 3 Million steps** (compared to PPO's 6293 in 10 Million steps). 

**If SAC was so much better, why wasn't it used for the rest of the project?**
1. **The Project Brief:** The core assignment strictly required the use of PPO for the expert to study imitation learning on an on-policy algorithm.
2. **The Normalization Incompatibility:** SAC uses a "replay buffer" to store past experiences and learn from them off-policy. As noted in the `PROJECT_OVERVIEW.md`, you cannot easily use `VecNormalize` with a replay buffer, because changing the running normalization statistics would corrupt all the old, buffered transitions. Therefore, the SAC agent had to be trained on *raw* observations. While this worked fine for the Ant, relying on raw observations makes Behavioural Cloning (Phase 3) much harder and more fragile, as the team proved in their E2 bonus experiments. Sticking to PPO allowed the team to use normalized observations consistently throughout the entire imitation pipeline.
