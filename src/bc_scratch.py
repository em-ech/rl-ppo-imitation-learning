"""From-scratch Behavioural Cloning in PyTorch (spec Section 5.3, M4).

StudentPolicy supports the architecture sweep (M8): variable hidden sizes and an
optional residual/skip-connection variant. train_bc returns the trained model
plus train/val loss curves for the learning-curve plot.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class StudentPolicy(nn.Module):
    """MLP mapping observation -> action. Tanh activations match the expert."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256),
                 skip: bool = False):
        super().__init__()
        self.skip = skip
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, act_dim)
        # Skip path needs matching dims; project obs to the body's output width.
        self.proj = nn.Linear(obs_dim, in_dim) if skip else None

    def forward(self, obs):
        h = self.body(obs)
        if self.skip:
            h = h + self.proj(obs)
        return self.head(h)


def train_bc(obs: np.ndarray, acts: np.ndarray, *, seed: int = 0,
             hidden=(256, 256), skip: bool = False, n_epochs: int = 50,
             batch_size: int = 256, lr: float = 1e-4, device: str = "cpu",
             val_frac: float = 0.1, normalize: bool = False):
    """Train a StudentPolicy by MSE regression to expert actions.

    normalize=True standardises observations to zero mean / unit variance (E2);
    the fitted mean/std are returned so deployment can reuse them.
    """
    torch.manual_seed(seed)
    obs = np.asarray(obs, dtype=np.float32)
    acts = np.asarray(acts, dtype=np.float32)

    obs_mean = obs.mean(axis=0, keepdims=True) if normalize else None
    obs_std = obs.std(axis=0, keepdims=True) + 1e-8 if normalize else None
    if normalize:
        obs = (obs - obs_mean) / obs_std

    ds = TensorDataset(torch.from_numpy(obs), torch.from_numpy(acts))
    n_val = int(val_frac * len(ds))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = StudentPolicy(obs.shape[1], acts.shape[1], hidden=hidden,
                          skip=skip).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    train_losses, val_losses = [], []
    for _ in range(n_epochs):
        model.train()
        running = 0.0
        for ob, ac in train_loader:
            ob, ac = ob.to(device), ac.to(device)
            loss = crit(model(ob), ac)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(ob)
        train_losses.append(running / max(n_train, 1))

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for ob, ac in val_loader:
                ob, ac = ob.to(device), ac.to(device)
                vloss += crit(model(ob), ac).item() * len(ob)
        val_losses.append(vloss / max(n_val, 1))

    return model, {"train": train_losses, "val": val_losses,
                   "obs_mean": obs_mean, "obs_std": obs_std}
