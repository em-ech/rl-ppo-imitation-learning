"""Shared PPO hyperparameter profiles (used by train_expert.py and pretraining.py).

`default` is the SB3-default config with a linear LR schedule (Walker2d). `tuned_ant`
is the rl-zoo3 Optuna-tuned Ant config. See DECISIONS.md for why these differ by
environment.
"""
from __future__ import annotations


def linear_schedule(initial: float):
    """LR decaying linearly from `initial` to 0 over training (progress 1 -> 0)."""
    return lambda progress_remaining: progress_remaining * initial


PROFILES = {
    "default": dict(n_steps=2048, batch_size=64, n_epochs=10,
                    learning_rate=linear_schedule(3e-4), clip_range=0.2,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5,
                    max_grad_norm=0.5),
    "tuned_ant": dict(n_steps=512, batch_size=32, n_epochs=10,
                      learning_rate=1.90609e-05, clip_range=0.1, gamma=0.98,
                      gae_lambda=0.8, ent_coef=4.9646e-07, vf_coef=0.677239,
                      max_grad_norm=0.6),
    "tuned_walker": dict(n_steps=512, batch_size=32, n_epochs=20,
                         learning_rate=5.05041e-05, clip_range=0.1, gamma=0.99,
                         gae_lambda=0.95, ent_coef=0.000585045, vf_coef=0.871923,
                         max_grad_norm=1.0),
}
