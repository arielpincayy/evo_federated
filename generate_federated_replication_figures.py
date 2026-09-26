#!/usr/bin/env python3
"""Generate multiseed publication figures from the FedAvg replication."""
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(os.environ.get("EVOFEDERATED_ROOT", "results/federated_fedavg_replication"))
OUT = Path("Paper/figures/federated_campaign")
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
}


def ci_half(values):
    values = pd.Series(values, dtype=float).dropna().to_numpy()
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))


def aggregate(df, group_column="method"):
    rows = []
    for method, group in df.groupby(group_column, sort=False):
        row = {group_column: method, "n": len(group)}
        for column in ["test_mean_accuracy", "test_min_accuracy", "test_mean_f1", "test_min_f1", "elapsed"]:
            row[column] = group[column].mean()
            row[f"{column}_ci"] = ci_half(group[column])
        row["n_local_trainings_search"] = group["n_local_trainings_search"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def grouped_quality(data, labels, colors, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.36
    for ax, mean_col, min_col, ylabel in (
        (axes[0], "test_mean_accuracy", "test_min_accuracy", "Test accuracy"),
        (axes[1], "test_mean_f1", "test_min_f1", "Test macro-F1"),
    ):
        ax.bar(
            x - width / 2,
            data[mean_col],
            width,
            yerr=data[f"{mean_col}_ci"],
            capsize=3,
            label="Mean",
            color=colors,
            edgecolor="#222222",
            linewidth=0.35,
        )
        ax.bar(
            x + width / 2,
            data[min_col],
            width,
            yerr=data[f"{min_col}_ci"],
            capsize=3,
            label="Minimum",
            color=colors,
            alpha=0.45,
            hatch="//",
            edgecolor="#222222",
            linewidth=0.35,
        )
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylim(0, 0.9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=11)
    save(fig, name)


def optimizer_hypervolume(summary):
    methods = ["nsga2", "nsga3", "moead"]
    labels = ["NSGA-II (2 objectives)", "NSGA-III (8 objectives)", "MOEA/D (8 objectives)"]
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["purple"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True)
    for ax, method, label, color in zip(axes, methods, labels, colors):
        seeds = sorted(summary[(summary.phase == "phase1_optimizers") & (summary.method == method)].seed.unique())
        records = []
        for seed in seeds:
            run = ROOT / "phase1_optimizers" / method / f"seed_{int(seed)}"
            hv = pd.read_csv(run / "generations" / "hypervolume.csv")
            hv["seed"] = int(seed)
            records.append(hv[["eval_count", "hv", "seed"]])
        data = pd.concat(records, ignore_index=True)
        grouped = data.groupby("eval_count")["hv"].agg(["mean", "std", "count"]).reset_index()
        grouped["ci"] = grouped.apply(
            lambda row: stats.t.ppf(0.975, row["count"] - 1) * row["std"] / np.sqrt(row["count"])
            if row["count"] > 1 else 0.0,
            axis=1,
        )
        ax.plot(grouped["eval_count"], grouped["mean"], marker="o", color=color, linewidth=2)
        ax.fill_between(
            grouped["eval_count"],
            grouped["mean"] - grouped["ci"],
            grouped["mean"] + grouped["ci"],
            color=color,
            alpha=0.18,
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Cumulative architecture evaluations")
        ax.set_ylabel("Internal hypervolume")
        ax.grid(color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Optimizer hypervolume trajectories across three seeds", fontsize=11)
    save(fig, "optimizer_hypervolume")


def participation_hypervolume(summary):
    methods = ["nsga2", "random-k", "similarity", "dissimilarity", "prob-similarity", "prob-dissimilarity"]
    labels = ["Full", "Random-4", "Similarity", "Dissimilarity", "Prob. similarity", "Prob. dissimilarity"]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"], PALETTE["purple"], PALETTE["gray"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for method, label, color in zip(methods, labels, colors):
        phase = "phase1_optimizers" if method == "nsga2" else "phase2_participation"
        seeds = sorted(summary[(summary.phase == phase) & (summary.method == method)].seed.unique())
        records = []
        for seed in seeds:
            run_method = "nsga2" if method == "nsga2" else method
            run_phase = "phase1_optimizers" if method == "nsga2" else "phase2_participation"
            run = ROOT / run_phase / run_method / f"seed_{int(seed)}"
            hv = pd.read_csv(run / "generations" / "hypervolume.csv")
            hv["seed"] = int(seed)
            records.append(hv[["eval_count", "hv", "seed"]])
        data = pd.concat(records, ignore_index=True)
        grouped = data.groupby("eval_count")["hv"].agg(["mean", "std", "count"]).reset_index()
        grouped["ci"] = grouped.apply(
            lambda row: stats.t.ppf(0.975, row["count"] - 1) * row["std"] / np.sqrt(row["count"])
            if row["count"] > 1 else 0.0,
            axis=1,
        )
        ax.plot(grouped["eval_count"], grouped["mean"], marker="o", linewidth=1.7, label=label, color=color)
        ax.fill_between(
            grouped["eval_count"],
            grouped["mean"] - grouped["ci"],
            grouped["mean"] + grouped["ci"],
            color=color,
            alpha=0.10,
        )
    ax.set_xlabel("Cumulative architecture evaluations")
    ax.set_ylabel("Internal hypervolume")
    ax.set_title("NSGA-II participation hypervolume")
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "participation_hypervolume")


def search_dynamics(summary):
    run_paths = [
        ROOT / "phase1_optimizers" / "nsga2" / f"seed_{int(seed)}"
        for seed in sorted(summary[summary.method == "nsga2"].seed.unique())
    ]
    hv_records = []
    client_records = []
    for run in run_paths:
        hv = pd.read_csv(run / "generations" / "hypervolume.csv")
        hv["seed"] = int(run.name.split("_")[1])
        hv_records.append(hv)
        pop = pd.read_csv(run / "generations" / "population_history.csv")
        best = pop.loc[pop.groupby("generation")["mean_accuracy"].idxmax()]
        for _, row in best.iterrows():
            for cid in range(8):
                client_records.append({
                    "seed": int(run.name.split("_")[1]),
                    "generation": int(row["generation"]),
                    "client": cid,
                    "accuracy": float(row[f"acc_client_{cid}"]),
                })
    hv = pd.concat(hv_records, ignore_index=True)
    hv_group = hv.groupby("eval_count")["hv"].agg(["mean", "std", "count"]).reset_index()
    hv_group["ci"] = hv_group.apply(lambda row: stats.t.ppf(0.975, row["count"] - 1) * row["std"] / np.sqrt(row["count"]), axis=1)
    clients = pd.DataFrame(client_records)
    client_group = clients.groupby(["generation", "client"])["accuracy"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)
    axes[0].plot(hv_group.eval_count, hv_group["mean"], marker="o", color=PALETTE["blue"], linewidth=2)
    axes[0].fill_between(hv_group.eval_count, hv_group["mean"] - hv_group.ci, hv_group["mean"] + hv_group.ci, color=PALETTE["blue"], alpha=0.18)
    axes[0].set_xlabel("Cumulative architecture evaluations")
    axes[0].set_ylabel("Internal hypervolume")
    axes[0].set_title("NSGA-II validation trajectory")
    axes[0].grid(color="#dddddd", linewidth=0.7)
    axes[0].set_axisbelow(True)
    for cid in range(8):
        part = client_group[client_group.client == cid]
        axes[1].plot(part.generation, part.accuracy, marker="o", linewidth=1.4, label=f"C{cid}")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title("Best-by-mean candidate per generation")
    axes[1].set_xticks(sorted(client_group.generation.unique()))
    axes[1].set_ylim(0.45, 0.9)
    axes[1].grid(color="#dddddd", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].legend(ncol=2, frameon=False, fontsize=7)
    fig.suptitle("Search dynamics across three seeds", fontsize=11)
    save(fig, "search_dynamics")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "campaign_summary.csv")

    optimizer_order = ["fixed", "nsga2", "random", "nsga3", "moead"]
    optimizer_labels = ["Fixed", "NSGA-II", "Random", "NSGA-III", "MOEA/D"]
    optimizer = aggregate(summary[summary.phase == "phase1_optimizers"].query("method in @optimizer_order"))
    optimizer["order"] = optimizer["method"].map({method: i for i, method in enumerate(optimizer_order)})
    optimizer = optimizer.sort_values("order").reset_index(drop=True)
    grouped_quality(
        optimizer,
        optimizer_labels,
        [PALETTE["gray"], PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]],
        "optimizer_quality",
        "Full participation: optimizer comparison (three seeds)",
    )

    participation_order = ["nsga2", "random-k", "similarity", "dissimilarity", "prob-similarity", "prob-dissimilarity"]
    participation_labels = ["Full", "Random-4", "Similarity", "Dissimilarity", "Prob. similarity", "Prob. dissimilarity"]
    participation_source = pd.concat([
        summary[(summary.phase == "phase1_optimizers") & (summary.method == "nsga2")],
        summary[summary.phase == "phase2_participation"],
    ], ignore_index=True)
    participation = aggregate(participation_source)
    participation["order"] = participation["method"].map({method: i for i, method in enumerate(participation_order)})
    participation = participation.sort_values("order").reset_index(drop=True)
    grouped_quality(
        participation,
        participation_labels,
        [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"], PALETTE["purple"], PALETTE["gray"]],
        "participation_quality",
        "NSGA-II: participation comparison at k=4 (three seeds)",
    )

    optimizer_hypervolume(summary)
    participation_hypervolume(summary)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)
    markers = ["o", "s", "^", "D", "P", "X"]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"], PALETTE["purple"], PALETTE["gray"]]
    for order, row in participation.iterrows():
        label = participation_labels[order]
        axes[0].errorbar(
            row["n_local_trainings_search"] + (order - 2.5) * 2.5,
            row["test_mean_accuracy"],
            yerr=row["test_mean_accuracy_ci"],
            color=colors[order], marker=markers[order], markersize=6,
            linestyle="none", capsize=3, label=label,
        )
        axes[1].barh(label, row["elapsed"], xerr=row["elapsed_ci"], color=colors[order], capsize=3)
    axes[0].set_xlabel("Search local trainings")
    axes[0].set_ylabel("Mean test accuracy")
    axes[0].set_xlim(0, max(500, float(participation["n_local_trainings_search"].max()) * 1.15))
    axes[0].set_ylim(0.65, 0.82)
    axes[0].grid(color="#dddddd", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.7)
    axes[1].set_axisbelow(True)
    fig.suptitle("Quality and measured cost under client selection (three seeds)", fontsize=11)
    save(fig, "participation_cost_quality")

    search_dynamics(summary)

    pd.DataFrame({
        "figure": ["optimizer_quality", "participation_quality", "participation_cost_quality", "optimizer_hypervolume", "participation_hypervolume", "search_dynamics"],
        "raw_data": [
            str(ROOT / "campaign_summary.csv"),
            str(ROOT / "campaign_summary.csv"),
            str(ROOT / "campaign_summary.csv"),
            str(ROOT / "phase1_optimizers/{nsga2,nsga3,moead}/seed_*/generations/hypervolume.csv"),
            str(ROOT / "{phase1_optimizers/nsga2,phase2_participation/*}/seed_*/generations/hypervolume.csv"),
            str(ROOT / "phase1_optimizers/nsga2/seed_*/generations"),
        ],
        "uncertainty": [
            "95% t interval across three seeds",
            "95% t interval across three seeds",
            "95% t interval across three seeds",
            "95% t interval across three seeds within each objective space",
            "95% t interval across three seeds within the common two-objective space",
            "left panel: 95% t interval; right panel: mean only",
        ],
        "transformation": [
            "mean/min metrics grouped by optimizer",
            "mean/min metrics grouped by participation method",
            "mean accuracy and elapsed time grouped by participation method",
            "internal hypervolume grouped by optimizer and evaluation count; objective spaces kept separate",
            "internal hypervolume grouped by participation method and evaluation count",
            "mean hypervolume and mean client trajectories across seeds",
        ],
    }).to_csv(OUT / "figure_provenance.csv", index=False)


if __name__ == "__main__":
    main()
