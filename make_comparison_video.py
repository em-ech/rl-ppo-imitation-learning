"""Multi-policy side-by-side comparison video for the comparison table.

For one environment, renders a deterministic episode of every available policy
(PPO expert, library BC, DAgger, SAC) and stacks them horizontally into a single
mp4 with a title bar above each panel. PPO/BC/DAgger consume the expert's
VecNormalize observations; SAC uses raw observations (it trains without
VecNormalize). Ablation rows (E1/E2) and the pretraining-stage runs are not single
deployable policies, so they are not rendered. Shorter rollouts are frozen on
their last frame so panels stay aligned.
Usage: make_comparison_video.py [ENV_ID]
"""
import sys

import gymnasium as gym

from src import config, eval, seeding, video
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.policies import ActorCriticPolicy

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Ant-v4"
seeding.set_seed(0)
VN_PATH = config.MODELS_DIR / f"ppo_expert_{ENV_ID}" / "vecnormalize.pkl"
norm = eval.load_obs_normalizer(VN_PATH)
N = (lambda x: norm(x)) if norm is not None else (lambda x: x)
print(f"[cmp-video] env={ENV_ID} normalized={norm is not None}", flush=True)


def build_panels():
    """List of (label, predict_fn) for the policies that exist for this env, in
    comparison-table order. Each loader is wrapped so one missing/broken policy
    does not abort the whole video."""
    panels = []

    def add(label, loader):
        try:
            panels.append((label, loader()))
            print(f"[cmp-video] loaded {label}", flush=True)
        except Exception as e:
            print(f"[cmp-video] skip {label}: {type(e).__name__}: {e}", flush=True)

    ppo_path = config.MODELS_DIR / f"ppo_expert_{ENV_ID}" / "best_model.zip"
    if ppo_path.exists():
        def ppo_fn():
            m = PPO.load(str(ppo_path.with_suffix("")), device="cpu")
            return lambda o: m.predict(N(o), deterministic=True)[0]
        add("PPO expert", ppo_fn)

    bc_path = config.MODELS_DIR / "bc_student" / f"bc_student_imitation_{ENV_ID}"
    if bc_path.exists():
        def bc_fn():
            p = ActorCriticPolicy.load(str(bc_path), device="cpu")
            return lambda o: p.predict(N(o), deterministic=True)[0]
        add("Library BC", bc_fn)

    dagger_path = config.MODELS_DIR / "dagger_student" / f"dagger_student_{ENV_ID}"
    if dagger_path.exists():
        def dagger_fn():
            p = ActorCriticPolicy.load(str(dagger_path), device="cpu")
            return lambda o: p.predict(N(o), deterministic=True)[0]
        add("DAgger", dagger_fn)

    # SAC trains without VecNormalize -> raw observations. Prefer the preserved
    # 3M snapshot for Ant so an in-progress extension does not change the clip.
    sac_dir = config.MODELS_DIR / f"sac_expert_{ENV_ID}"
    sac_path = sac_dir / "best_model_3M.zip"
    if not sac_path.exists():
        sac_path = sac_dir / "best_model.zip"
    if sac_path.exists():
        def sac_fn():
            m = SAC.load(str(sac_path.with_suffix("")), device="cpu")
            return lambda o: m.predict(o, deterministic=True)[0]
        add("SAC", sac_fn)

    return panels


def rollout(predict, seed=0, max_steps=1000):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, ret = [], 0.0
    for _ in range(max_steps):
        frames.append(env.render())
        obs, r, term, trunc, _ = env.step(predict(obs))
        ret += r
        if term or trunc:
            break
    env.close()
    return frames, ret


panels = build_panels()
if not panels:
    print(f"[cmp-video] no policies found for {ENV_ID}; nothing to render", flush=True)
    sys.exit(0)

rolls = [(label, *rollout(fn)) for label, fn in panels]
for label, frames, ret in rolls:
    print(f"[cmp-video] {label:11s} return {ret:7.0f}  ({len(frames)} frames)", flush=True)

length = min(max(len(fr) for _, fr, _ in rolls), 1000)


def pad(frames):
    return frames[:length] if len(frames) >= length else frames + [frames[-1]] * (length - len(frames))


tracks = [video.titled_track(pad(fr), f"{label}  ({ret:.0f})") for label, fr, ret in rolls]
combined = video.stack_panels(tracks)

try:
    out = config.VIDEOS_DIR / f"comparison_{ENV_ID}.mp4"
    video.write_mp4(combined, out)
    labels = " | ".join(label for label, _, _ in rolls)
    print(f"[cmp-video] DONE -> {out} ({length} frames; {labels})", flush=True)
except Exception as e:
    print(f"[cmp-video] FAILED: {type(e).__name__}: {e}", flush=True)
