#!/usr/bin/env python3
"""Generate publication figures from the controlled FedAvg campaign only."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/federated_fedavg_campaign")
OUT = Path("Paper/figures/federated_campaign")
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def grouped_quality(data, labels, colors, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.36
    for ax, mean_col, min_col, ylabel in (
        (axes[0], "test_mean_accuracy", "test_min_accuracy", "Test accuracy"),
        (axes[1], "test_mean_f1", "test_min_f1", "Test macro-F1"),
    ):
        ax.bar(x - width / 2, data[mean_col], width, label="Mean", color=colors, edgecolor="#222222", linewidth=0.35)
        ax.bar(x + width / 2, data[min_col], width, label="Minimum", color=colors, alpha=0.45,
               hatch="//", edgecolor="#222222", linewidth=0.35)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylim(0, 0.85)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=11)
    save(fig, name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "campaign_summary.csv")

    optimizer_order = ["fixed", "nsga2", "random", "nsga3", "moead"]
    optimizer_labels = ["Fixed", "NSGA-II", "Random", "NSGA-III", "MOEA/D"]
    optimizer = summary.set_index("method").loc[optimizer_order].reset_index()
    optimizer_colors = [PALETTE["gray"], PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]]
    grouped_quality(
        optimizer,
        optimizer_labels,
        optimizer_colors,
        "optimizer_quality",
        "Full participation: optimizer comparison (one seed)",
    )

    participation_order = ["nsga2", "random-k", "similarity", "dissimilarity", "prob-similarity", "prob-dissimilarity"]
    participation_labels = ["Full", "Random-k", "Similarity", "Dissimilarity", "Prob. similarity", "Prob. dissimilarity"]
    participation = pd.concat([
        summary[(summary.phase == "phase1_optimizers") & (summary.method == "nsga2")],
        summary[(summary.phase == "phase2_participation")],
    ], ignore_index=True)
    participation["order"] = participation["method"].map({name: i for i, name in enumerate(participation_order)})
    participation = participation.sort_values("order").reset_index(drop=True)
    participation_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"], PALETTE["purple"], PALETTE["gray"]]
    grouped_quality(
        participation,
        participation_labels,
        participation_colors,
        "participation_quality",
        "NSGA-II: participation comparison (one seed)",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    markers = ["o", "s", "^", "D", "P", "X"]
    for _, row in participation.iterrows():
        order = int(row["order"])
        label = participation_labels[order]
        color = participation_colors[order]
        jittered_cost = row["n_local_trainings_search"] + (order - 2.5) * 2.5
        axes[0].scatter(jittered_cost, row["test_mean_accuracy"], s=65, color=color,
                        marker=markers[order], label=label)
        axes[1].barh(label, row["elapsed"], color=color)
    axes[0].set_xlabel("Search local trainings")
    axes[0].set_ylabel("Mean test accuracy")
    axes[0].set_xlim(0, 215)
    axes[0].set_ylim(0.66, 0.76)
    axes[0].grid(color="#dddddd", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.7)
    axes[1].set_axisbelow(True)
    fig.suptitle("Quality and measured cost under client selection (one seed)", fontsize=11)
    save(fig, "participation_cost_quality")

    hv = pd.read_csv(ROOT / "phase1_optimizers/nsga2/seed_42/generations/hypervolume.csv")
    pop = pd.read_csv(ROOT / "phase1_optimizers/nsga2/seed_42/generations/population_history.csv")
    best = pop.loc[pop.groupby("generation")["mean_accuracy"].idxmax()].sort_values("generation")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    axes[0].plot(hv["eval_count"], hv["hv"], marker="o", color=PALETTE["blue"], linewidth=2)
    axes[0].set_xlabel("Cumulative architecture evaluations")
    axes[0].set_ylabel("Internal hypervolume")
    axes[0].set_title("NSGA-II search trajectory")
    axes[0].grid(color="#dddddd", linewidth=0.7)
    axes[0].set_axisbelow(True)
    for cid in range(8):
        axes[1].plot(best["generation"], best[f"acc_client_{cid}"], marker="o", linewidth=1.4, label=f"C{cid}")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title("Best-by-mean candidate per generation")
    axes[1].set_xticks(best["generation"])
    axes[1].set_ylim(0.5, 0.9)
    axes[1].grid(color="#dddddd", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].legend(ncol=2, frameon=False, fontsize=7)
    fig.suptitle("Search dynamics and client heterogeneity (one seed)", fontsize=11)
    save(fig, "search_dynamics")

    pd.DataFrame({
        "figure": ["optimizer_quality", "participation_quality", "participation_cost_quality", "search_dynamics"],
        "raw_data": [
            str(ROOT / "campaign_summary.csv"),
            str(ROOT / "campaign_summary.csv"),
            str(ROOT / "campaign_summary.csv"),
            str(ROOT / "phase1_optimizers/nsga2/seed_42/generations/population_history.csv"),
        ],
        "uncertainty": ["not estimated; one seed"] * 4,
        "transformation": ["mean/min metrics by persisted run"] * 3 + ["best validation-mean row per generation"],
    }).to_csv(OUT / "figure_provenance.csv", index=False)


if __name__ == "__main__":
    main()
