# Extensions Deep Dive: Bonuses & Ablations

In addition to the core 5-phase imitation learning pipeline, the team tackled several bonus extensions to deeply understand the mechanics of Behavioural Cloning and test alternative Reinforcement Learning architectures.

Here is a deep dive into the three major extensions: the Noisy Expert (E1), the Normalization Ablation (E2), and the Off-Policy Soft Actor-Critic (SAC) experiment.

---

## 1. Extended Requirement E1: The Noisy Expert

**The Goal:** Test how "fragile" the Behavioural Cloning student is. If the expert provides slightly bad or noisy data, can the student still figure out the underlying pattern and learn to walk?

**The Process (`noise_sweep.py`):** 
The team didn't train a new, bad PPO agent. Instead, they took the pristine dataset collected in Phase 2 and mathematically injected random Gaussian noise directly into the recorded actions. They tested increasing levels of noise standard deviation (from $\sigma = 0.0$ up to $\sigma = 0.8$), clipped the corrupted actions to the valid torque range (`[-1.0, 1.0]`), and trained BC students on these noisy datasets across 5 different random seeds.

**The Finding:** The two environments reacted completely differently.
* **Walker2d is highly fragile:** Even at the tiniest amount of noise ($\sigma = 0.05$), the Walker2d student immediately collapsed, dropping below 50% of the expert's performance. By $\sigma=0.8$, it only retained 5% of the performance. 
* **Ant is incredibly robust:** The Ant student barely noticed the noise. It held between 95% and 102% of the expert's performance all the way up to $\sigma = 0.4$, and only started to bend (down to 71%) at the maximum tested noise of $\sigma = 0.8$. It never collapsed entirely.

---

## 2. Extended Requirement E2: Observation Normalization Ablation

**The Goal:** Prove scientifically whether observation normalization (`VecNormalize`) is strictly necessary for Behavioural Cloning, or if the neural network can just figure out the raw scales itself.

**The Process (`norm_ablation.py`):**
The team trained the from-scratch PyTorch BC student using the exact same dataset, but under two conditions:
1. **Normalised:** The training observations were standardized to a zero mean and unit variance using the dataset's own statistics.
2. **Raw:** The network was fed the massive, unscaled numbers directly from the MuJoCo engine.

**The Finding:** Once again, the environments split perfectly along the lines of fragility.
* **Walker2d demands normalization:** Without normalization, the Walker2d student completely failed to converge. The normalized student scored `4654`, while the raw student scored `1163` (a massive **4.0x gap**).
* **Ant is mathematically invincible:** The Ant student performed almost identically whether the data was normalized (`5679`) or raw (`5946`). The difference was within the margin of random seed noise.

> [!NOTE]
> **The Grand Theme of E1 & E2**
> These ablations definitively answer Research Question 6 (RQ6). Many researchers assume that because Ant has more dimensions (27) and is a 3D robot, it is "harder" than the 2D Walker. The team proved the exact opposite. Walker2d is a highly fragile, saturated control problem that desperately needs clean labels and normalized inputs. Ant is highly robust to imperfect imitation. *Imitation difficulty is completely disconnected from state dimensionality.*

---

## 3. Bonus Experiment: Off-Policy SAC

**The Goal:** The core brief mandated the use of PPO (an on-policy algorithm). The team wanted to experiment beyond the brief and test an off-policy algorithm (Soft Actor-Critic, or SAC) to see if it was more sample-efficient.

**The Process (`train_sac.py`):**
SAC uses a "replay buffer" to store past experiences and learn from them off-policy. The team trained SAC agents using the `rl-zoo3` MuJoCo recipes for 3 Million steps. 

> [!WARNING]
> **The `VecNormalize` Incompatibility**
> The team documented a critical engineering limitation: You cannot easily use `VecNormalize` with SAC. Because SAC trains from old memories stored in a replay buffer, if the running normalization statistics change over time, the old buffered observations become mathematically corrupted. Therefore, SAC was strictly trained on *raw* observations.

**The Finding:** SAC absolutely crushed PPO in terms of sample efficiency.
* **Ant-v4:** While the PPO expert took 10 Million steps (and heavy tuning) to reach a score of `6293`, the SAC agent rocketed past that, achieving a massive score of **`7295` in just 3 Million steps**.
* **HalfCheetah-v4 (Off-Brief):** The team also tested SAC on the HalfCheetah environment to see if they could hit the professor's "stretch goal" of 8000. SAC obliterated the stretch goal, scoring **`15387` in 3 Million steps**.
