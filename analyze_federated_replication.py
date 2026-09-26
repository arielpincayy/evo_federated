#!/usr/bin/env python3
"""Summarize the paired multiseed FedAvg-NAS replication."""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(os.environ.get("EVOFEDERATED_ROOT", "results/federated_fedavg_replication"))
METRICS = [
    "test_mean_accuracy",
    "test_min_accuracy",
    "test_mean_f1",
    "test_min_f1",
]
COSTS = [
    "n_local_trainings_search",
    "n_local_trainings_total",
    "communication_bytes_search",
    "communication_bytes_final",
    "elapsed",
]


def mean_ci(values):
    values = pd.Series(values, dtype=float).dropna().to_numpy()
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if len(values) > 1:
        half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    else:
        half = 0.0
    return mean, sd, mean - half, mean + half


def summarize_group(df, phase, method):
    rows = df[(df["phase"] == phase) & (df["method"] == method)]
    record = {"phase": phase, "method": method, "n": int(len(rows))}
    for column in METRICS + COSTS:
        mean, sd, low, high = mean_ci(rows[column])
        record.update({
            f"{column}_mean": mean,
            f"{column}_sd": sd,
            f"{column}_ci95_low": low,
            f"{column}_ci95_high": high,
        })
    return record


def paired_rows(df, phase, baseline, methods, baseline_phase=None):
    baseline_phase = baseline_phase or phase
    base = df[(df["phase"] == baseline_phase) & (df["method"] == baseline)].set_index("seed")
    rows = []
    for method in methods:
        current = df[(df["phase"] == phase) & (df["method"] == method)].set_index("seed")
        common = sorted(set(base.index) & set(current.index))
        for seed in common:
            row = {
                "phase": phase,
                "baseline": f"{baseline_phase}/{baseline}",
                "method": method,
                "seed": int(seed),
            }
            for column in METRICS + COSTS:
                row[f"delta_{column}"] = float(current.loc[seed, column] - base.loc[seed, column])
            rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(paired):
    rows = []
    for (phase, baseline, method), group in paired.groupby(["phase", "baseline", "method"]):
        row = {"phase": phase, "baseline": baseline, "method": method, "n": len(group)}
        for column in [c for c in paired.columns if c.startswith("delta_")]:
            mean, sd, low, high = mean_ci(group[column])
            row.update({
                f"{column}_mean": mean,
                f"{column}_sd": sd,
                f"{column}_ci95_low": low,
                f"{column}_ci95_high": high,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def validate_inputs(df):
    expected = {
        ("phase1_optimizers", method): 3
        for method in ["fixed", "random", "nsga2", "nsga3", "moead"]
    }
    expected.update({
        ("phase2_participation", method): 3
        for method in ["random-k", "similarity", "dissimilarity", "prob-similarity", "prob-dissimilarity"]
    })
    observed = df.groupby(["phase", "method"]).size().to_dict()
    if observed != expected:
        raise ValueError(f"Unexpected run layout: {observed}")
    if set(df["search_split"]) != {"val"} or set(df["final_split"]) != {"test"}:
        raise ValueError("Unexpected train/validation/test split semantics")
    if set(df["final_participation"]) != {"full"}:
        raise ValueError("Unexpected final participation semantics")


def main():
    summary_path = ROOT / "campaign_summary.csv"
    df = pd.read_csv(summary_path)
    validate_inputs(df)

    phase1_methods = ["fixed", "random", "nsga2", "nsga3", "moead"]
    phase2_methods = ["random-k", "similarity", "dissimilarity", "prob-similarity", "prob-dissimilarity"]
    grouped = pd.DataFrame(
        [
            *[summarize_group(df, "phase1_optimizers", method) for method in phase1_methods],
            *[summarize_group(df, "phase2_participation", method) for method in phase2_methods],
        ]
    )
    paired = pd.concat(
        [
            paired_rows(df, "phase1_optimizers", "nsga2", ["random", "nsga3", "moead"]),
            paired_rows(df, "phase2_participation", "nsga2", phase2_methods, "phase1_optimizers"),
        ],
        ignore_index=True,
    )
    paired_agg = paired_summary(paired)
    grouped.to_csv(ROOT / "method_summary_multiseed.csv", index=False)
    paired.to_csv(ROOT / "paired_differences.csv", index=False)
    paired_agg.to_csv(ROOT / "paired_summary.csv", index=False)

    with open(ROOT / "analysis_metadata.json", "w") as handle:
        json.dump({
            "source": str(summary_path),
            "seeds": sorted(int(seed) for seed in df["seed"].unique()),
            "n_runs": int(len(df)),
            "metrics": METRICS,
            "costs": COSTS,
            "ci": "two-sided t interval around the mean; descriptive with n=3",
            "primary_candidate": "campaign_summary uses primary_candidate from each final_test.csv",
        }, handle, indent=2)

    print(f"Validated {len(df)} runs")
    print(grouped[["phase", "method", "n", "test_mean_accuracy_mean", "test_mean_accuracy_sd", "n_local_trainings_search_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
