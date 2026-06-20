# Phase 2 Deep Dive: Expert Demonstration Collection

Once the PPO expert is fully trained and verified, the next step is to create a "textbook" for the student agent to learn from. This textbook is a dataset of recorded observations (what the robot sees) and actions (what the robot does).

Here is a deep dive into how the team built this dataset, the methodological choices they made, and how they improved upon the standard guidelines.

## 1. The Process
The team implemented the collection process in `collect_demos.py` and `src/collect.py`. The process is straightforward but heavily optimized:
1. Load the "best" PPO expert saved from Phase 1.
2. Load the environment (Walker2d or Ant).
3. Run the expert in the environment for 100 full episodes. *(Note: The guidelines specified a minimum of 50 episodes, but also explicitly required an ablation study up to 100 episodes in Phase 3. The team efficiently generated the full 100 upfront to satisfy both requirements).*
4. At every single timestep, record the `observation` the robot sees, and the `action` the expert decides to take.
5. Save these as massive NumPy arrays (`observations.npy`, `actions.npy`, `episode_returns.npy`) inside the `data/` directory.

## 2. Key Methodological Points (Following Guidelines)

The team followed several strict requirements from the project guidelines to ensure the dataset was pristine:

* **Deterministic Actions:** In PPO, the agent's policy is technically a Gaussian distribution (it outputs a mean and a standard deviation, and samples randomly from it to explore). When collecting demonstrations, the team strictly used `deterministic=True`. This forces the expert to output the *exact mean* of the distribution (its absolute best guess) without any random exploration noise. This provides a much cleaner, less noisy signal for the student to learn from.
* **Strict Quality Gates:** Before saving the dataset, the script runs a final check: *Do at least 90% of these 100 recorded episodes achieve a return higher than two-thirds of the average?* If the expert falls over too often, the script flags a `FAIL`. As noted in Phase 1, the Ant environment struggled to pass this gate until the `tuned_ant` profile was used. *(For context: according to the `PROJECT_OVERVIEW.md`, an early Ant expert that scored 2850 failed this gate because only 68% of its episodes passed, and its minimum return was a disastrous 42—meaning it fell over quickly. It wasn't until the team tuned the hyperparameters to reach a score of 6293 that it consistently passed the 90% gate).*
* **Exploratory Data Analysis (EDA):** As required by Minimum Requirement M2 (detailed in Section 7.1 of the project guidelines PDF), the script automatically generates data visualization plots (`outputs/dataset_analysis_[ENV].png`) to check the distribution of actions per joint and the return distribution across episodes. This acts as a sanity check before training the student.
* **The Normalization Requirement:** As we saw in Phase 1, the Ant expert was trained using `VecNormalize`. Because testing the necessity of normalization was an explicit extended requirement (E2), handling it correctly here was a strict methodological necessity. The script mathematically normalizes the observation *right before* handing it to the expert so it can predict correctly, but it **saves the RAW, unnormalized observation to the numpy array**. This ensures the team had the raw data required to run the E2 ablation studies in Phase 3.

## 3. What Makes This Implementation Stand Out? (The Differentiators)

While the guidelines provided a basic script to collect data (Listing 4 in the PDF), the team heavily engineered their `src/collect.py` module to fix a major flaw in the original guideline script.

> [!TIP]
> **Preserving Episode Boundaries (`episode_starts.npy`)**
> *The Problem:* The PDF guidelines suggest saving the dataset as one giant, flat array of observations and actions. However, in Phase 3, you have to run an "ablation study" where you train the student on smaller subsets of data (e.g., 5 episodes, 10 episodes). If you just slice a flat array (e.g., `data[:1000]`), you will slice the data right in the middle of a robot's jump, corrupting the sequence.
> *The Fix:* The team engineered their collection script to explicitly save `episode_lengths.npy` and `episode_starts.npy`. They wrote a custom `subset()` function that uses these indices to perfectly slice the dataset cleanly at the boundaries of whole episodes, ensuring the ablation studies are mathematically sound.


## 4. Final Results of Phase 2
* **Walker2d:** 100 complete episodes collected. Passed the 90% quality gate easily.
* **Ant:** 100 complete episodes collected. Passed the 90% quality gate (after the PPO hyperparameters were tuned to 10M steps).
* **Data Volume:** Each environment generated massive NumPy arrays containing tens of thousands of transitions, cleanly indexed by episode boundaries, ready for Behavioural Cloning.
