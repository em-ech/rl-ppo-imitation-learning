"""Shared figure helpers so the four notebooks do not duplicate plotting code."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def learning_curves(train, val, title="Behavioural Cloning Learning Curves"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train, label="Train MSE")
    ax.plot(val, label="Validation MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def dataset_eda(returns, actions, lengths=None):
    """EDA panels: return distribution, per-joint action distribution, and
    (if lengths given) the trajectory-length histogram (spec 5.3 / 9.2)."""
    n = 3 if lengths is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    axes[0].hist(returns, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Expert Episode Return Distribution")
    axes[0].set_xlabel("Return"); axes[0].set_ylabel("Count")
    for j in range(actions.shape[1]):
        axes[1].hist(actions[:, j], bins=40, alpha=0.4, label=f"Joint {j}")
    axes[1].set_title("Action Distribution per Joint")
    axes[1].set_xlabel("Torque"); axes[1].legend(fontsize=7)
    if lengths is not None:
        axes[2].hist(lengths, bins=20, color="seagreen", edgecolor="white")
        axes[2].set_title("Trajectory Length Histogram")
        axes[2].set_xlabel("Episode length (steps)"); axes[2].set_ylabel("Count")
    return fig


def epoch_curve(epochs, returns, expert_mean, title="BC Return vs Training Epochs"):
    """Library-BC performance as a function of training epochs (M3)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, returns, marker="o", color="steelblue", label="BC student")
    ax.axhline(expert_mean, color="green", linestyle="--",
               label=f"Expert ({expert_mean:.0f})")
    ax.set_xlabel("Training Epochs"); ax.set_ylabel("Mean Evaluation Return")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def ablation(episode_counts, means, stds, expert_mean, title="BC Performance vs Expert Data Volume"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(episode_counts, means, yerr=stds, marker="o", capsize=5,
                color="steelblue", label="BC student")
    ax.axhline(expert_mean, color="green", linestyle="--",
               label=f"Expert ({expert_mean:.0f})")
    ax.set_xlabel("Number of Expert Episodes")
    ax.set_ylabel("Mean Evaluation Return")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def arch_bars(names, means, stds, expert_mean, title="BC Architecture Sweep"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", edgecolor="white")
    ax.axhline(expert_mean, color="green", linestyle="--",
               label=f"Expert ({expert_mean:.0f})")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Mean Evaluation Return"); ax.set_title(title)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    return fig


def noise_curve(sigmas, means, stds, expert_mean,
                title="BC Robustness to Noisy Expert Actions (E1)"):
    """BC return as a function of Gaussian action-noise std (E1)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sigmas, means, yerr=stds, marker="o", capsize=5,
                color="firebrick", label="BC student")
    ax.axhline(expert_mean, color="green", linestyle="--",
               label=f"Expert ({expert_mean:.0f})")
    # Half-of-expert reference makes the collapse point easy to read off.
    ax.axhline(0.5 * expert_mean, color="grey", linestyle=":",
               label="50% of expert")
    ax.set_xlabel("Action-noise std (sigma, on [-1, 1] torque scale)")
    ax.set_ylabel("Mean Evaluation Return")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def norm_ablation(curves, finals, expert_mean,
                  title="Observation Normalisation Ablation (E2)"):
    """Convergence + final-return comparison of BC with vs without obs
    normalisation (E2). curves maps label -> validation-loss list; finals maps
    label -> (mean, std) evaluation return."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, val in curves.items():
        axes[0].plot(val, label=label)
    axes[0].set_title("Validation MSE vs Epoch (convergence speed)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation MSE")
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    names = list(finals)
    x = np.arange(len(names))
    means = [finals[n][0] for n in names]
    stds = [finals[n][1] for n in names]
    axes[1].bar(x, means, yerr=stds, capsize=5, color="steelblue", edgecolor="white")
    axes[1].axhline(expert_mean, color="green", linestyle="--",
                    label=f"Expert ({expert_mean:.0f})")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Mean Evaluation Return")
    axes[1].set_title("Final Return (deployed)")
    axes[1].legend(); axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def sample_efficiency(curves, target=None, title="SAC vs PPO: Sample Efficiency"):
    """Eval return vs environment timesteps for several algorithms (bonus SAC
    experiment). curves maps label -> (timesteps array, returns array)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (steps, returns) in curves.items():
        ax.plot(steps, returns, marker="o", markersize=3, label=label)
    if target is not None:
        ax.axhline(target, color="grey", linestyle=":", label=f"target ({target:.0f})")
    ax.set_xlabel("Environment timesteps"); ax.set_ylabel("Mean Evaluation Return")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def bc_vs_dagger(bc_by_epoch, dagger_by_iter, expert_mean):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(bc_by_epoch, color="steelblue", label="BC student")
    axes[0].axhline(expert_mean, color="green", linestyle="--",
                    label=f"Expert ({expert_mean:.0f})")
    axes[0].set_title("Behavioural Cloning: Return vs Epochs")
    axes[0].set_xlabel("Training Epochs"); axes[0].set_ylabel("Mean Evaluation Return")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(dagger_by_iter, color="darkorange", marker="o", label="DAgger student")
    axes[1].axhline(expert_mean, color="green", linestyle="--",
                    label=f"Expert ({expert_mean:.0f})")
    axes[1].axhline(np.mean(bc_by_epoch[-5:]), color="steelblue", linestyle=":",
                    label="BC (converged)")
    axes[1].set_title("DAgger: Return vs Iteration")
    axes[1].set_xlabel("DAgger Iteration"); axes[1].set_ylabel("Mean Evaluation Return")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle("BC vs DAgger: Imitation Learning Comparison", fontsize=13, fontweight="bold")
    return fig
