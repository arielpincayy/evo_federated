#!/usr/bin/env python3
"""Ejecuta experimento completo."""
import argparse
import sys
from pathlib import Path

# Ensure src in path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evofederated.utils.config import load_config
from evofederated.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run EvoFederated experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml/json config")
    parser.add_argument("--output", type=str, default=None, help="Override output dir")
    parser.add_argument("--verbose", action="store_true", help="Verbose pymoo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output) if args.output else None
    res = run_experiment(cfg, output_dir=out, verbose=args.verbose)
    print(f"\nExperiment finished. Results at: {res['output_dir']}")
    print(res["summary"])


if __name__ == "__main__":
    main()
