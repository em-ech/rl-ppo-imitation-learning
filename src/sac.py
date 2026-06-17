"""Shared SAC hyperparameter profiles (bonus: off-policy expert experiment).

SAC is off-policy and maximum-entropy, so unlike the PPO setup it uses a replay
buffer, automatic entropy tuning, and no VecNormalize (running reward/obs stats
would corrupt transitions already stored in the buffer). `default` is plain SAC;
`tuned_ant` / `tuned_halfcheetah` are the rl-zoo3 MuJoCo recipe (gSDE exploration
and batched updates), which reach the strong published returns. See README.
"""
from __future__ import annotations

# rl-zoo3 MuJoCo SAC recipe shared across the tuned profiles (net_arch differs).
_MUJOCO_SAC = dict(learning_rate=7.3e-4, buffer_size=1_000_000, batch_size=256,
                   tau=0.02, gamma=0.98, train_freq=8, gradient_steps=8,
                   learning_starts=10_000, ent_coef="auto", use_sde=True)


SAC_PROFILES = {
    "default": dict(learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
                    tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                    learning_starts=10_000, ent_coef="auto",
                    policy_kwargs=dict(net_arch=[256, 256])),
    "tuned_ant": dict(**_MUJOCO_SAC,
                      policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300])),
    "tuned_halfcheetah": dict(**_MUJOCO_SAC,
                              policy_kwargs=dict(log_std_init=-3, net_arch=[256, 256])),
}
