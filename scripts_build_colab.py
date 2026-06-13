"""Generate colab/colab_runner.ipynb: a one-stop notebook to run the whole
pipeline on Google Colab with Drive persistence. Run: python scripts_build_colab.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "colab"
OUT.mkdir(exist_ok=True)


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


cells = [
    md("# Colab Runner: PPO + Imitation Learning\n\n"
       "Runs the full pipeline on Colab. Code is cloned from GitHub; trained "
       "models and results are persisted to Google Drive so disconnects do not "
       "lose work.\n\n"
       "**Compute note:** PPO training is CPU-bound (MuJoCo physics), so a GPU "
       "runtime does not speed up the expert; it only helps from-scratch BC. "
       "Use a GPU runtime for the BC ablation/sweep, CPU is fine for PPO. Long "
       "runs need Colab Pro or an active tab to avoid idle disconnects."),
    md("## 1. Clone the code from GitHub"),
    code("REPO = 'https://github.com/em-ech/rl-ppo-imitation-learning.git'\n"
         "CODE_DIR = '/content/GroupProject'\n"
         "import os, sys, shutil\n"
         "os.chdir('/content')  # never stand inside the dir we are about to delete\n"
         "if os.path.exists(CODE_DIR):\n"
         "    shutil.rmtree(CODE_DIR)\n"
         "!git clone -q {REPO} {CODE_DIR}\n"
         "assert os.path.isdir(CODE_DIR), 'clone failed'\n"
         "sys.path.insert(0, CODE_DIR)\n"
         "os.chdir(CODE_DIR)\n"
         "print('code in', CODE_DIR, '->', sorted(os.listdir(CODE_DIR))[:8])"),
    md("## 2. Mount Drive for persistence"),
    code("from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "DRIVE_ROOT = '/content/drive/MyDrive/rl_project'\n"
         "os.makedirs(DRIVE_ROOT, exist_ok=True)\n"
         "print('Drive root:', DRIVE_ROOT)"),
    md("## 3. Install pinned dependencies\n\n"
       "This pins numpy 1.26 (required by gymnasium 0.29, which imitation 1.0 "
       "forces), downgrading Colab's preinstalled numpy 2.x. **After this cell "
       "finishes you MUST restart the runtime** (Runtime -> Restart session) for "
       "the downgrade to take effect, then re-run cells 1-2 and continue from "
       "cell 4. Skip this cell on the second pass (packages persist across a "
       "restart). The long 'dependency resolver' conflict warnings about "
       "jax/transformers/opencv/torchvision are harmless: this project does not "
       "use those packages."),
    code("!pip -q install -r /content/GroupProject/requirements.txt\n"
         "print('\\nInstall done. NOW: Runtime -> Restart session, then re-run '\n"
         "      'cells 1-2 and continue from cell 4.')"),
    md("## 4. Configure persistence + sanity check\n\n"
       "Pointing `PROJECT_DATA_ROOT` at Drive sends models/, data/, outputs/, "
       "logs/ to Drive so they survive disconnects. `os.environ` changes are "
       "inherited by the `!python ...` calls below."),
    code("import sys, os  # re-assert in case the runtime was restarted\n"
         "sys.path.insert(0, '/content/GroupProject'); os.chdir('/content/GroupProject')\n"
         "os.environ['PROJECT_DATA_ROOT'] = DRIVE_ROOT\n"
         "from src import config\n"
         "print('device:', config.device())\n"
         "print('models ->', config.MODELS_DIR)\n"
         "import gymnasium as gym\n"
         "for e in ['Walker2d-v4', 'Ant-v4']:\n"
         "    gym.make(e).reset(seed=0); print('ok', e)"),
    md("## 5. Stage 1 - PPO experts (M1)\n\n"
       "Walker2d with the validated config. For Ant, the faithful rl-zoo3 "
       "Optuna-tuned profile (`tuned_ant`, n_envs=1, ~1e7 steps) is the best shot "
       "at the 4000 target but is slow (roughly 18h on Colab). If the run is "
       "interrupted, re-run the same cell with `resume` appended: it continues "
       "from the last 100k checkpoint on Drive instead of restarting. Each run "
       "writes best_model (and vecnormalize.pkl for Ant) to Drive."),
    code("!cd /content/GroupProject && python train_expert.py Walker2d-v4 4000000 4"),
    code("# Faithful rl-zoo3 tuned Ant (slow). Append 'resume' to continue after a disconnect.\n"
         "!cd /content/GroupProject && python train_expert.py Ant-v4 10000000 1 norm tuned_ant\n"
         "# resume:\n"
         "# !cd /content/GroupProject && python train_expert.py Ant-v4 10000000 1 norm tuned_ant resume\n"
         "# Faster fallback (our config, ~8h, reached ~2850): \n"
         "# !cd /content/GroupProject && python train_expert.py Ant-v4 16000000 8 norm"),
    md("## 6. Stage 2 - demonstration collection + EDA + quality gate (M2)"),
    code("!cd /content/GroupProject && python collect_demos.py Walker2d-v4 100\n"
         "!cd /content/GroupProject && python collect_demos.py Ant-v4 100"),
    md("## 7. Stage 3 - BC experiments (M3, M4, M5) and multi-seed arch sweep (M8)\n\n"
       "Use a GPU runtime here for the from-scratch BC runs."),
    code("!cd /content/GroupProject && python bc_experiments.py Walker2d-v4\n"
         "!cd /content/GroupProject && python arch_sweep.py Walker2d-v4\n"
         "# repeat for Ant once its expert and dataset are ready:\n"
         "# !cd /content/GroupProject && python bc_experiments.py Ant-v4\n"
         "# !cd /content/GroupProject && python arch_sweep.py Ant-v4"),
    md("## 8. Stages 4-5 - DAgger and pretraining\n\n"
       "Run `notebooks/04_dagger.ipynb` and `notebooks/05_pretraining.ipynb` "
       "(still in development). All outputs already persist to Drive via "
       "`PROJECT_DATA_ROOT`."),
    md("## Recovering after a disconnect\n\n"
       "Models, checkpoints (every 100k steps), datasets, and figures live under "
       "`MyDrive/rl_project/`. Re-run cells 1-4 to remount and reattach, then "
       "continue. Note: `train_expert.py` restarts a run from scratch rather than "
       "resuming mid-training, so prefer Colab Pro or keep the tab active for the "
       "long expert runs."),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

(OUT / "colab_runner.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", OUT / "colab_runner.ipynb")
