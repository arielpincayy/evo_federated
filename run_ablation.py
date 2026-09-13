#!/usr/bin/env python3
"""Lanza estudios de ablation variando ejes configurables.
Cada ablation genera múltiples experimentos secuenciales.
No paraleliza por defecto para evitar OOM, pero puede extenderse.
"""
import argparse
import sys
import itertools
import copy
from pathlib import Path
import json
import dataclasses

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evofederated.utils.config import load_config
from evofederated.experiments.runner import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--axes", type=str, nargs="*", default=None, help="Axes to ablate: alpha k tau metric seed. If None, run all combos small.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Try to load raw yaml to get ablation section
    import yaml, pathlib
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    ablation = raw.get("ablation", {})

    # Build combos
    combos = []
    # Example: if user wants to ablate only alpha, etc.
    # We'll generate cartesian product of requested axes if provided, else do single-axis sweeps
    # For simplicity, generate single-parameter sweeps (not full cartesian) to control cost.

    output_base = Path(cfg.output_dir)
    experiments = []

    # Alpha sweep
    alphas = ablation.get("alphas", [])
    ks = ablation.get("k_epochs", [])
    metrics = ablation.get("distance_metrics", [])
    taus = ablation.get("taus", [])

    # If specific axes requested, filter
    if args.axes:
        if "alpha" not in args.axes:
            alphas = []
        if "k" not in args.axes:
            ks = []
        if "metric" not in args.axes:
            metrics = []
        if "tau" not in args.axes:
            taus = []

    # Build experiments: each is a copy of cfg with one parameter changed (single sweep), not full factorial
    # This is more affordable.
    def add_experiments_for_param(param_name, values, setter):
        for v in values:
            c = copy.deepcopy(cfg)
            setter(c, v)
            # unique output dir
            c.output_dir = str(output_base / f"{param_name}_{v}")
            experiments.append((f"{param_name}={v}", c))

    add_experiments_for_param("alpha", alphas, lambda c,v: setattr(c.dataset, "alpha", v))
    add_experiments_for_param("k", ks, lambda c,v: setattr(c.characterization, "k_epochs", v))
    add_experiments_for_param("metric", metrics, lambda c,v: setattr(c.characterization, "distance_metric", v))
    add_experiments_for_param("tau", taus, lambda c,v: (setattr(c.characterization, "tau", v), setattr(c.evolution, "tau", v)))

    if not experiments:
        print("No ablation experiments generated. Check ablation config.")
        return

    print(f"Generated {len(experiments)} ablation experiments:")
    for name,_ in experiments:
        print(" -", name)
    if args.dry_run:
        print("Dry run, not executing.")
        return

    for name, c in experiments:
        print(f"\n=== Ablation: {name} ===")
        # Derive output dir
        out = Path(c.output_dir)
        res = run_experiment(c, output_dir=out)
        print(f"Done {name} -> {out}")

if __name__ == "__main__":
    main()
