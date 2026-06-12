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


def dataset_eda(returns, actions):
    """Return-distribution and per-joint action-distribution panels (spec 5.3)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(returns, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Expert Episode Return Distribution")
    axes[0].set_xlabel("Return"); axes[0].set_ylabel("Count")
    for j in range(actions.shape[1]):
        axes[1].hist(actions[:, j], bins=40, alpha=0.4, label=f"Joint {j}")
    axes[1].set_title("Action Distribution per Joint")
    axes[1].set_xlabel("Torque"); axes[1].legend(fontsize=7)
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
