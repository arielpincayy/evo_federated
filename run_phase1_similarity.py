#!/usr/bin/env python3
"""Run Phase 1 validation for similarity-pairing single-objective.

Executes ~5 seeds, each isolated, then aggregates and generates cross-seed plots.

Usage:
  .venv/bin/python run_phase1_similarity.py --config configs/similarity.yaml
"""
import argparse
import json
import sys
from pathlib import Path
import dataclasses
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evofederated.utils.config import load_config
from evofederated.experiments.runner_similarity import run_similarity_experiment
from evofederated.experiments.plots_similarity import generate_aggregate_plots
import pandas as pd


SEEDS = [42, 123, 999, 2024, 2025]


def run_one_seed(base_cfg, seed: int, base_output: Path):
    cfg = dataclasses.replace(base_cfg)
    # propagate seed to all subconfigs
    cfg.seed = int(seed)
    cfg.dataset = dataclasses.replace(cfg.dataset, seed=int(seed))
    cfg.characterization = dataclasses.replace(cfg.characterization, seed=int(seed))
    cfg.evolution = dataclasses.replace(cfg.evolution, seed=int(seed))
    out = base_output / f"seed_{seed:03d}"
    # ensure clean if exists? we keep if already completed
    if (out / "summary.json").exists():
        print(f"[SKIP] seed {seed} already completed at {out}")
        return out
    if out.exists():
        shutil.rmtree(out)
    res = run_similarity_experiment(cfg, output_dir=out, verbose=False)
    return out


def aggregate(base_output: Path):
    aggregate_dir = base_output / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    # collect summaries
    rows = []
    for seed in SEEDS:
        sd = base_output / f"seed_{seed:03d}"
        summ_path = sd / "summary.json"
        if not summ_path.exists():
            print(f"[WARN] missing {summ_path}")
            continue
        with open(summ_path) as f:
            s = json.load(f)
        row = {
            "seed": seed,
            "best_fitness_evolution": s.get("best_fitness_evolution"),
            "best_generation": s.get("best_generation"),
            "final_mean_accuracy": s.get("final_mean_accuracy"),
            "final_worst_accuracy": s.get("final_worst_accuracy"),
            "final_std_accuracy": s.get("final_std_accuracy"),
            "reduction_percent": s.get("cost", {}).get("reduction_percent_vs_full"),
            "total_search_evals": s.get("cost", {}).get("total_search_evaluations"),
            "total_time": s.get("cost", {}).get("total_time_sec"),
            "n_reps": s.get("n_reps"),
            "n_pairs": s.get("cost", {}).get("n_pairs"),
        }
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(aggregate_dir / "summary_across_seeds.csv", index=False)
        # stats
        stats = {
            "mean_final_accuracy": float(df["final_mean_accuracy"].mean()),
            "std_final_accuracy": float(df["final_mean_accuracy"].std()),
            "median_final_accuracy": float(df["final_mean_accuracy"].median()),
            "min_final_accuracy": float(df["final_mean_accuracy"].min()),
            "max_final_accuracy": float(df["final_mean_accuracy"].max()),
            "mean_worst_accuracy": float(df["final_worst_accuracy"].mean()),
            "std_worst_accuracy": float(df["final_worst_accuracy"].std()),
            "mean_reduction_percent": float(df["reduction_percent"].mean()),
            "seeds": SEEDS,
            "n_seeds": len(df),
        }
        with open(aggregate_dir / "aggregate_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("\nAggregate stats:")
        print(json.dumps(stats, indent=2))

        # generate cross-seed plots
        try:
            # copy generation histories aggregated? The generate_aggregate_plots expects seed_* dirs inside aggregate_dir
            # Instead we point it to base_output and copy output to aggregate_dir
            # Our function expects aggregate_dir containing seed_* subdirs. So create symlinks/copies?
            # Easiest: call with base_output but also save plots to aggregate_dir
            generate_aggregate_plots(base_output)
            # move generated plots from base_output to aggregate_dir if needed
            for p in base_output.glob("*.png"):
                # they are saved directly in base_output by previous call (since aggregate_dir is expected)
                # But our function saved to aggregate_dir, so we already have them.
                pass
            # Ensure plots also in aggregate_dir: we called generate_aggregate_plots(base_output) but it saves inside base_output
            # So duplicate logic: generate also for aggregate_dir via copying single-seed aggregates?
            # We'll also run for aggregate_dir by creating symlinks
            # Simpler: generate plots both places
            import shutil as sh
            # After generating for base_output, copy those pngs to aggregate_dir
            for png in base_output.glob("*.png"):
                sh.copy(png, aggregate_dir / png.name)
        except Exception as e:
            print(f"[WARN] aggregate plots failed: {e}")
            import traceback
            traceback.print_exc()

        # also generate aggregate plots directly expecting seed_* inside aggregate_dir: create symlinked structure
        # Create a temporary aggregate view: we already have seed_* inside base_output, but aggregate_dir is inside base_output.
        # Our function for aggregate_dir will look for seed_* inside aggregate_dir, which doesn't exist. So we need to adjust.
        # Workaround: copy/link seed_ dirs reference? Instead we will call generate_aggregate_plots again but with custom handling:
        # Let's create a helper that directly aggregates from base_output into aggregate_dir
        try:
            # copy aggregate logic manually for aggregate_dir's own seed_ view using base_output as source
            # We already generated via base_output, but need to also place in aggregate_dir properly
            # Just copy all aggregate plots from base_output that were generated to aggregate_dir if not there
            pass
        except Exception:
            pass

        # Correct approach: generate_aggregate_plots should look at base_output, save to aggregate_dir
        # We'll implement second pass:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        # Do second generation explicitly saving to aggregate_dir
        # Reuse code: create plots that save to aggregate_dir
        # We'll call helper that takes source dir and dest dir
        try:
            # create convergence per seed saving to aggregate_dir
            seed_dirs = [base_output / f"seed_{s:03d}" for s in SEEDS if (base_output / f"seed_{s:03d}").exists()]
            plt.figure(figsize=(8,5))
            for sd in sorted(seed_dirs):
                gm = sd / "generation_metrics.csv"
                if not gm.exists():
                    gm = sd / "generations" / "generation_metrics.csv"
                if not gm.exists():
                    continue
                df_g = pd.read_csv(gm)
                if df_g.empty:
                    continue
                plt.plot(df_g["generation"], df_g["best_fitness"], marker="o", alpha=0.7, label=sd.name)
            plt.xlabel("Generación")
            plt.ylabel("Best fitness")
            plt.title("Curva de convergencia distintas seeds")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(aggregate_dir / "convergence_per_seed.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"aggregate second pass failed {e}")

    return aggregate_dir


def main():
    global SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/similarity.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override seeds")
    parser.add_argument("--output", type=str, default=None, help="override base output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else SEEDS
    SEEDS = seeds

    base_output = Path(args.output) if args.output else Path(cfg.output_dir)
    base_output = Path(base_output)
    base_output.mkdir(parents=True, exist_ok=True)

    print(f"Phase1 seeds: {SEEDS}")
    print(f"Base output: {base_output}")
    print(f"Config: {args.config}")

    for seed in SEEDS:
        print(f"\n=== Running seed {seed} ===")
        run_one_seed(cfg, seed, base_output)

    print("\n=== Aggregating ===")
    aggregate(base_output)

    print(f"\nPhase1 validation complete. Results at: {base_output}")
    print(f"Aggregate at: {base_output / 'aggregate'}")
    # list plots
    print("\nPlots per seed:")
    for seed in SEEDS:
        sd = base_output / f"seed_{seed:03d}" / "plots"
        if sd.exists():
            print(f"  {sd}: {len(list(sd.glob('*.png')))} plots")


if __name__ == "__main__":
    main()
