# Phase 3 Deep Dive: Behavioural Cloning (BC)

In this phase, the team trained a "student" agent to mimic the expert using the demonstration dataset. In Behavioural Cloning, the student never interacts with the actual environment or sees the reward signal—it is treated purely as a supervised learning regression task (given state $S$, predict expert action $A$).

Here is a deep dive into the BC implementation, the massive ablation studies conducted, and the engineering details that set this repository apart.

## 1. The Process
The team implemented two parallel tracks for BC in `bc_experiments.py`:
1. **Library BC:** Using the official `imitation` Python library built on top of `stable-baselines3`.
2. **From-Scratch BC:** A custom PyTorch implementation (`src/bc_scratch.py`) written by the team to allow for custom architectures, early stopping, and deeper control.

The training process involved feeding the expert's observations into the student's Multi-Layer Perceptron (MLP) and minimizing the Mean Squared Error (MSE) between the student's predicted action and the expert's recorded action.

## 2. Key Methodological Points

* **The Normalization Bug (Crucial Fix):** As discussed in Phase 1, the Ant expert was trained using `VecNormalize`. This means the expert expects inputs like `0.5`, but the raw dataset saved in Phase 2 has raw inputs like `500.0`. If you train a BC student on the raw data, it completely fails to fit the curve. The team correctly engineered `bc_experiments.py` to load the expert's `vecnormalize.pkl` and mathematically transform the entire training dataset *before* training the student.
* **Epoch Budget & Early Stopping:** A common mistake in BC is undertraining. When the team first trained the Walker2d student for 50 epochs (the default baseline suggested in the project guidelines), they documented in the `PROJECT_OVERVIEW.md` that it severely underfit, only achieving 47% of the expert's performance. The team bumped the budget to 150 epochs. More importantly, they implemented **Validation-Loss Early Stopping** in their from-scratch PyTorch implementation. It monitors a 10% validation holdout set and saves the model weights at the exact epoch where validation MSE hits its minimum, preventing the student from overfitting to the demonstration data.
* **Offline vs Online Metrics:** The team tracked both *Offline* metrics (Validation MSE, which purely measures static mapping: given a pre-recorded picture, can you guess the joystick movement?) and *Online* metrics (Environment Return, which drops the agent into the live physics engine to see if it can actually walk). The team noted an interesting theoretical finding: **a low BC offline loss does not guarantee a good online policy**. Because the robot is evaluated sequentially in the live environment, tiny prediction errors compound over time. A student with a seemingly perfect MSE of `0.0005` might make one tiny mistake, end up in a strange new posture it has never seen before, and immediately fall over.
  *(Note: Early stopping and more epochs only delay this "compounding error" problem. The true mathematical fix for this is **DAgger** (Phase 4), which interleaves data collection and training so the student learns how to recover from its own mistakes!)*

## 3. What Makes This Implementation Stand Out? (The Differentiators)

The team went far above and beyond the minimum requirements by running rigorous, multi-seed ablation studies.

> [!TIP]
> **Rigorous Multi-Seed Ablation Studies (`arch_sweep.py`)**
> To answer the research questions, the team tested how much data is enough (`{5, 10, 20, 50, 100}` episodes) and whether network architecture matters (Small `[64,64]`, Default `[256,256]`, Large `[512,512]`, Skip Connections). 
> Instead of just running these once, **they ran every single configuration across 5 different random seeds**. This is the gold standard for RL research, allowing them to report accurate error bars rather than relying on a single "lucky" training run.

> [!NOTE]
> **Environment-Dependent Architecture Findings**
> The multi-seed sweep revealed a fascinating insight: 
> * For **Walker2d**, a large network (`[512,512]`) was absolutely necessary to capture the complex, near-saturated control dynamics. The small network severely underfit.
> * For **Ant**, the architecture barely mattered at all. All networks clustered tightly around the same high performance. This proved that Ant is a much easier environment to imitate, despite having more dimensions.

> [!IMPORTANT]
> **Device & Compute Management**
> The team discovered a bug where the `imitation` library would crash with a device mismatch on GPUs. They engineered their scripts to force the library BC onto the CPU (which is actually faster for tiny MLPs), while routing their custom from-scratch PyTorch BC to leverage GPU/Apple Metal Performance Shaders (MPS) for maximum speed.

## 4. Final Results of Phase 3
The Behavioural Cloning students successfully learned to walk purely by imitating the expert!

* **Walker2d:** The Library BC student recovered **~95%** of the expert's performance (Score: 5719 vs Expert's 6043).
* **Ant:** The Library BC student recovered an incredible **~99%** of the expert's performance (Score: 6237 vs Expert's 6293).

*(Bonus)* The team also ran Extended Requirements (E1 and E2) evaluating the student's robustness to noisy experts and unnormalized data. For the noisy expert (E1), they didn't train a new PPO agent; as per the guidelines, they simply injected artificial Gaussian noise into the recorded actions of the existing clean dataset, and tested if the student could still learn the underlying pattern. These tests proved that Walker2d is highly fragile to imperfect imitation, while Ant remains robust.
