# Phase 4 Deep Dive: DAgger (Dataset Aggregation)

In Phase 3, we discovered a fatal flaw with pure Behavioural Cloning: **Covariate Shift** (the compounding error problem). If the BC student makes one tiny mistake, it ends up in a strange posture it has never seen in the training data, panics, and falls over. 

Phase 4 implements the mathematical solution to this problem: **DAgger**.

Here is a deep dive into how DAgger works, how the team fixed broken code in the professor's guidelines, and the key engineering choices made.

## 1. The Process: How DAgger Fixes Covariate Shift
Instead of training purely offline on a static, pre-recorded dataset, DAgger is an *interactive* algorithm. It works in a loop:
1. **Rollout:** Drop the student into the live MuJoCo environment and let it try to walk.
2. **The Oracle:** The perfectly trained PPO Expert from Phase 1 watches the student in real-time. Even when the student makes a mistake and gets into a weird posture, the Oracle records what *it* would do to recover.
3. **Aggregate:** These new recovery states (the student's mistakes + the expert's corrective actions) are added to the massive dataset.
4. **Retrain:** The student is retrained on this new, aggregated dataset.

By explicitly teaching the student how to recover from its own mistakes, DAgger eliminates the compounding error problem entirely.

## 2. What Makes This Implementation Stand Out? (The Fixes)

The project guidelines provided a sample script (Listing 10 in the PDF) to run DAgger using the `imitation` library. However, the team discovered multiple critical bugs in the provided code and had to re-engineer the pipeline in `src/dagger.py`.

> [!WARNING]
> **Fixing the Broken Guidelines (Listing 10)**
> The code provided in the assignment PDF was incompatible with the modern version of the `imitation` library (v1.0.0). For example, it passed `None` to the BC trainer, which caused an immediate crash. The team successfully diagnosed the API changes and rewrote the `run_dagger()` function to correctly instantiate a proper `bc.BC` trainer and wire it to the `SimpleDAggerTrainer`.


> [!TIP]
> **CPU-Only Thread Management**
> The `imitation` library's DAgger implementation is notoriously buggy when running on GPUs due to device-mismatch errors between the vectorized environment and PyTorch tensors. The team explicitly forced the entire pipeline to run on `device="cpu"` and used `torch.set_num_threads(1)` to prevent CPU thread thrashing, ensuring rock-solid stability during long training runs.

## 3. Key Methodological Points

* **Iteration Budget:** The team ran DAgger for 12 iterations, gathering 5,000 steps per iteration. *(Why these numbers? The project guidelines explicitly suggested 12 iterations and 5000 steps as a solid benchmark. This yields 60,000 new, on-policy transitions collected over the course of the DAgger run).*
* **Training Budget (25 BC Epochs):** In each iteration, the student is trained for 25 epochs. *(Why 25? Early tests using fewer epochs severely underfit the data, making DAgger look artificially worse than BC. The team scaled it up to 25 epochs per iteration to ensure a "fair" comparison of total computational effort against the massive 100+ epoch budget given to the pure BC agents).*
* **The Normalization Wrapper Injection:** Just like in Phase 3, dealing with Normalization was a strict methodological requirement due to the E2 bonus experiment. If DAgger drops the student into a raw environment, but the queried Oracle (the Ant expert) expects normalized inputs, the Oracle will give garbage corrective actions, ruining the dataset. The team engineered a custom `make_venv()` function inside the DAgger loop that seamlessly wraps the live environment with the expert's `VecNormalize` statistics (locking them in `eval` mode with `venv.training = False` so they don't drift).

## 4. Final Results of Phase 4
* **Results:** As expected, DAgger drastically outperformed static Behavioural Cloning. Because DAgger actively targets the exact states where the student is confused, it is incredibly sample-efficient. The student quickly converged to (and even slightly exceeded) the expert's performance on both Walker2d and Ant, effectively solving the Imitation Learning task.

| Metric | Walker2d | Ant |
| :--- | :--- | :--- |
| **PPO Expert Baseline** | **6043** | **6293** |
| Behavioural Cloning (Phase 3) | 5719 (95%) | 6237 (99%) |
| **DAgger (Phase 4)** | **6208 (102%)** | **6564 (104%)** |
