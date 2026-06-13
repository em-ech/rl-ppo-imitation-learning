"""Behavioural Cloning via the imitation library, plus the BC -> PPO bridge.

train_bc_imitation trains an SB3 ActorCriticPolicy with imitation's BC. Because
that student *is* an SB3 policy, ppo_from_policy can load its weights into a
fresh PPO model to warm-start fine-tuning (spec Stage 5, the missing listing).
"""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from . import config


def _sb3_device(device: str) -> str:
    """Force CPU for the imitation library + SB3 paths.

    The imitation BC has a device-placement bug (demo tensors stay on CPU while
    the policy is on the accelerator) that breaks on both MPS and CUDA. These
    MLPs are tiny and CPU-bound anyway (SB3 itself advises CPU for MLP PPO), so
    we always run them on CPU. The from-scratch BC (bc_scratch) still uses the
    GPU.
    """
    return "cpu"


def build_transitions(obs: np.ndarray, acts: np.ndarray):
    """Wrap flat (obs, act) arrays as an imitation Transitions object.

    next_obs/dones are placeholders: pure BC ignores them (it regresses action
    on observation), so crossing episode boundaries here is harmless.
    """
    from imitation.data.types import Transitions

    obs = np.asarray(obs, dtype=np.float32)
    acts = np.asarray(acts, dtype=np.float32)
    n = len(obs)
    return Transitions(
        obs=obs[:-1],
        acts=acts[:-1],
        next_obs=obs[1:],
        dones=np.zeros(n - 1, dtype=bool),
        infos=np.array([{}] * (n - 1)),
    )


def train_bc_imitation(obs: np.ndarray, acts: np.ndarray, env_id: str, *,
                       seed: int = 0, n_epochs: int = 50, batch_size: int = 256,
                       lr: float = 1e-4, device: str = "cpu"):
    """Train a BC student on an SB3 policy. Returns (bc_trainer, env).

    The trained SB3 policy lives at bc_trainer.policy and can be handed to
    ppo_from_policy for Stage 5.
    """
    import gymnasium as gym
    from imitation.algorithms import bc

    device = _sb3_device(device)
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)

    student_policy = PPO(
        "MlpPolicy", env, seed=seed, device=device,
        policy_kwargs=dict(net_arch=config.NET_ARCH),
    ).policy

    trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=build_transitions(obs, acts),
        policy=student_policy,
        batch_size=batch_size,
        optimizer_kwargs={"lr": lr},
        device=device,
        rng=rng,
    )
    trainer.train(n_epochs=n_epochs, progress_bar=False)
    return trainer, env


def ppo_from_policy(trained_policy, env, *, seed: int = 0, device: str = "cpu",
                    **ppo_kwargs) -> PPO:
    """Create a PPO model warm-started from a BC/DAgger-trained policy.

    The actor weights transfer; the value head retrains during PPO. Use this for
    the 'BC + PPO fine-tune' and 'DAgger + PPO fine-tune' Stage 5 conditions.
    """
    device = _sb3_device(device)
    model = PPO(
        "MlpPolicy", env, seed=seed, device=device,
        policy_kwargs=dict(net_arch=config.NET_ARCH),
        **ppo_kwargs,
    )
    model.policy.load_state_dict(trained_policy.state_dict(), strict=False)
    return model
