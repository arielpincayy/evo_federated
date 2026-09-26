#!/usr/bin/env python3
"""Run and summarize the controlled FedAvg NAS campaign."""
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evofederated.experiments.runner_federated import run_federated_nas
from evofederated.utils.config import load_config


# The pilot remains under results/federated_fedavg_campaign. Replications can
# be directed to a separate root without overwriting prior artifacts.
SEEDS = tuple(int(value) for value in os.environ.get("EVOFEDERATED_SEEDS", "42").split(","))
PHASE1 = ("nsga2", "random", "nsga3", "moead", "fixed")
PHASE2 = ("full", "random-k", "similarity", "dissimilarity", "prob-similarity", "prob-dissimilarity")


def configure(seed):
    cfg = load_config("configs/federated_campaign.yaml")
    cfg.seed = seed
    cfg.dataset.seed = seed
    cfg.characterization.seed = seed
    cfg.evolution.seed = seed
    return cfg


def run_one(cfg, phase, label, optimizer, output_dir):
    output_dir = Path(output_dir)
    marker = output_dir / "final_evaluation" / "final_test.csv"
    if marker.exists():
        print(f"[SKIP] {phase}/{label}/seed_{cfg.seed}", flush=True)
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print(f"[RUN] {phase}/{label}/seed_{cfg.seed}", flush=True)
    result = run_federated_nas(
        cfg,
        strategy=label if phase == "phase2" else "full",
        optimizer=optimizer,
        tau=cfg.characterization.tau,
        output_dir=output_dir,
        verbose=False,
    )
    print(json.dumps(result["cost"], sort_keys=True), flush=True)


def main():
    root = Path(os.environ.get("EVOFEDERATED_ROOT", "results/federated_fedavg_campaign"))
    rows = []
    for seed in SEEDS:
        cfg = configure(seed)
        for optimizer in PHASE1:
            out = root / "phase1_optimizers" / optimizer / f"seed_{seed}"
            run_one(cfg, "phase1", optimizer, optimizer, out)
        for strategy in PHASE2:
            if strategy == "full":
                # Identical to phase1/nsga2/full; reuse that controlled run.
                continue
            out = root / "phase2_participation" / strategy / f"seed_{seed}"
            run_one(cfg, "phase2", strategy, "nsga2", out)

    for path in sorted(root.glob("phase*/*/seed_*/cost.json")):
        cost = json.loads(path.read_text())
        final_path = path.parent / "final_evaluation" / "final_test.csv"
        if not final_path.exists():
            continue
        final = pd.read_csv(final_path)
        primary = final[final.get("primary_candidate", False)] if "primary_candidate" in final else final.iloc[:1]
        if primary.empty:
            primary = final.iloc[:1]
        primary = primary.iloc[0]
        rows.append({
            "phase": path.parents[2].name,
            "method": path.parents[1].name,
            "seed": int(path.parent.name.split("_")[1]),
            **cost,
            "test_mean_accuracy": float(primary["test_mean_accuracy"]),
            "test_min_accuracy": float(primary["test_min_accuracy"]),
            "test_mean_f1": float(primary["test_mean_f1"]),
            "test_min_f1": float(primary["test_min_f1"]),
        })
    summary = pd.DataFrame(rows).sort_values(["phase", "method", "seed"])
    summary.to_csv(root / "campaign_summary.csv", index=False)
    summary.groupby(["phase", "method"], as_index=False).agg(
        seeds=("seed", "count"),
        test_mean_accuracy=("test_mean_accuracy", "mean"),
        test_min_accuracy=("test_min_accuracy", "mean"),
        test_mean_f1=("test_mean_f1", "mean"),
        test_min_f1=("test_min_f1", "mean"),
        n_arch_evals=("n_arch_evals", "mean"),
        n_local_trainings_search=("n_local_trainings_search", "mean"),
        elapsed=("elapsed", "mean"),
    ).to_csv(root / "campaign_summary_by_method.csv", index=False)
    print(f"Wrote {root / 'campaign_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
