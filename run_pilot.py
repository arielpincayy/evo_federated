#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evofederated.utils.config import load_config
from evofederated.experiments.pilot import run_pilot

def main():
    parser = argparse.ArgumentParser(description="Run pilot study")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-archs", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output) if args.output else None
    res = run_pilot(cfg, output_dir=out, n_archs=args.n_archs)
    print("Pilot done at", res["output_dir"])

if __name__ == "__main__":
    main()
