# Research Questions: Answers & Evidence

The team successfully answered all of the research questions (RQ1-RQ6, plus the central thesis) laid out in the project guidelines. The final, consolidated answers are located in Section 4 of the `PROJECT_OVERVIEW.md` file.

Here is a comprehensive breakdown of the answers, the specific experiments that proved them, and the outputs (notebooks/figures) generated as evidence.

---

### **Central Question: Can imitation reduce PPO's sample complexity?**
**The Answer:** Yes, decisively.
**The Explanation:** Training PPO from scratch takes millions of steps. By spending a tiny fraction of compute upfront on Behavioural Cloning (BC) or DAgger, you can warm-start a PPO agent and bypass the random exploration phase entirely.
**The Evidence:** Evaluated at 1.5 Million environment steps:
* PPO from scratch is still learning (Walker2d: ~1.1k, Ant: ~5.0k).
* Imitation-pretrained PPO is already at/near expert level (Walker2d: ~5.7k, Ant: ~6.6k).
**Where to find the outputs:** Phase 5 (`pretraining.py`). Output graph: `outputs/pretraining_[ENV].png`.

---

### **RQ1: How close can BC get to the expert, and what limits the gap?**
**The Answer:** Very close (95% for Walker2d, 99% for Ant). 
**The Explanation:** The dominant limiter initially was simply the training budget (the student was undertraining). The *residual* gap (the last 1%–5% it couldn't quite reach) is caused by the compounding error problem (covariate shift)—the student makes tiny prediction mistakes that compound over time in the live environment.
**The Evidence:** The epoch sweep showed performance climbing steadily from 1 epoch up to 100+ epochs.
**Where to find the outputs:** Phase 3 (`bc_experiments.py`) and Notebook `03_behavioural_cloning.ipynb`. Output graph: `outputs/bc_epoch_sweep_[ENV].png`.

---

### **RQ2: How does the amount of expert data affect the student?**
**The Answer:** Performance rises steeply from 5 to 50 episodes, then flatlines (saturates).
**The Explanation:** You don't need infinite data to learn these environments. Just 5-10 episodes yielded poor, high-variance results across different random seeds. However, around 50 episodes, the student captured almost all the achievable performance. Giving it 100 episodes didn't yield much more benefit.
**The Evidence:** The massive 5-seed dataset-size ablation study.
**Where to find the outputs:** Phase 3 (`bc_experiments.py`) and Notebook `03_behavioural_cloning.ipynb`. Output graph: `outputs/bc_ablation_data_size_[ENV].png`.

---

### **RQ3: Does lower BC loss imply a better policy?**
**The Answer:** No. 
**The Explanation:** Offline Mean Squared Error (MSE) measures static prediction accuracy, while Online Return measures dynamic walking ability. Because the robot walks sequentially, one tiny error can lead to an unrecoverable posture. A model with a "perfect" low MSE might make one fatal mistake and score poorly, while a model with a slightly higher MSE might make mistakes that don't cause the robot to fall over.
**The Evidence:** Comparing the Validation MSE vs Evaluation Return across the multi-seed training runs.
**Where to find the outputs:** Documented in the Phase 3 outputs and Notebook `03_behavioural_cloning.ipynb`.

---

### **RQ4: How much does student architecture matter?**
**The Answer:** It is highly environment-dependent.
**The Explanation:** 
* On Walker2d, architecture matters massively. A tiny network (`[64,64]`) underfits terribly, while a large network (`[512,512]`) is required to capture the complex control dynamics.
* On Ant, architecture barely matters. All networks—small or large—clustered tightly around the same high performance (MSE ~5e-4).
**The Evidence:** The 5-seed architecture sweep.
**Where to find the outputs:** Phase 3 (`arch_sweep.py`) and Notebook `03_behavioural_cloning.ipynb`. Output graph: `outputs/bc_arch_sweep_[ENV].png`.

---

### **RQ5: Does DAgger reduce covariate shift vs plain BC?**
**The Answer:** Yes, clearly.
**The Explanation:** When given a "fair" computational budget, DAgger matches or beats BC on every single configuration. You can see the mechanism working in real-time on the Walker2d graph: the performance climbs steadily from ~600 to ~6200 iteration by iteration as DAgger aggregates on-policy states and the expert labels recovery behaviors.
**The Evidence:** The DAgger iteration curves plotted side-by-side against the BC epoch baseline.
**Where to find the outputs:** Phase 4 (`dagger_run.py`) and Notebook `04_dagger.ipynb`. Output graph: `outputs/bc_vs_dagger_[ENV].png`.

---

### **RQ6: What are the systematic differences between Walker2d and Ant?**
**The Answer:** Ant is harder to *train*, but easier to *imitate*. Walker2d is easier to *train*, but harder to *imitate*.
**The Explanation:** 
* Many assume Ant is harder to imitate because it's a 3D robot with more joints (27 dimensions). The team proved this is false. Ant is incredibly robust to noisy data, missing normalization, and small neural networks.
* Walker2d is highly fragile. It relies on aggressive, near-saturated control. If the student makes a tiny mistake, the Walker falls over instantly. It requires perfect normalization, zero noise, and huge neural networks to imitate successfully.
**The Evidence:** The cross-environment results compiled from the entire pipeline, plus the E1 (Noise) and E2 (Normalization) ablations.
**Where to find the outputs:** The conclusions are drawn together in Notebook `05_pretraining.ipynb` and Notebook `06_extended.ipynb`. Output graphs: `outputs/bc_noise_sweep_[ENV].png` and `outputs/bc_norm_ablation_[ENV].png`.
