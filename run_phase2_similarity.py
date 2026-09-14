#!/usr/bin/env python3
"""Phase 2 benchmark for similarity pairing variant.

Implements experiments A-J per spec, saves under
results/similarity_pairing/phase2_experiments/

- A: deterministic similarity (max-weight matching, alternating)
- B: random pairing
- C: probabilistic similarity with tau sweep
- D: N scaling [4,8,16,32]
- E: heterogeneity alpha [0.1,0.5,1.0,10.0] (+ 0.3)
- F: K exploratory epochs [1,2,5,10]
- G: rep alternating vs random
- H: full evaluation baseline
- I: previous algorithm (dissimilarity + NSGA-II) si posible
- J: repeatability already via 10 seeds

Parallel via ProcessPoolExecutor (spawn) to utilize workstation.
"""
import argparse
import json
import sys
import dataclasses
import shutil
from pathlib import Path
import concurrent.futures
import multiprocessing
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evofederated.utils.config import load_config

BASE_CFG_PATH = "configs/similarity.yaml"
BASE_OUTPUT = Path("results/similarity_pairing/phase2_experiments")

SEEDS_10 = [42, 123, 999, 2024, 2025, 7, 11, 13, 17, 19]
SEEDS_5 = [42, 123, 999, 2024, 2025]
SEEDS_3 = [42, 123, 999]

TAU_VALUES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
N_VALUES = [4, 8, 16, 32]
ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0, 10.0]
K_VALUES = [1, 2, 5, 10]

def clone_cfg(base, seed: int):
    cfg = dataclasses.replace(base)
    cfg.seed = int(seed)
    cfg.dataset = dataclasses.replace(cfg.dataset, seed=int(seed))
    cfg.characterization = dataclasses.replace(cfg.characterization, seed=int(seed))
    cfg.evolution = dataclasses.replace(cfg.evolution, seed=int(seed))
    return cfg

def run_single_task(args):
    """Worker for parallel execution. args is dict with config overrides."""
    # Set multiprocessing spawn safe: limit torch threads
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    # Import inside worker to avoid pickling issues
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from evofederated.experiments.runner_similarity import run_similarity_experiment
    from evofederated.utils.config import load_config
    import dataclasses
    # args contains: base_cfg_path, output, seed, overrides dict, pairing_mode, tau, rep_mode, full_mode
    base_cfg = load_config(args["base_cfg_path"])
    seed = args["seed"]
    cfg = dataclasses.replace(base_cfg)
    cfg.seed = int(seed)
    cfg.dataset = dataclasses.replace(cfg.dataset, seed=int(seed))
    cfg.characterization = dataclasses.replace(cfg.characterization, seed=int(seed))
    cfg.evolution = dataclasses.replace(cfg.evolution, seed=int(seed))
    # apply overrides
    ov = args.get("overrides", {})
    if "n_clients" in ov:
        cfg.dataset = dataclasses.replace(cfg.dataset, n_clients=int(ov["n_clients"]))
    if "alpha" in ov:
        cfg.dataset = dataclasses.replace(cfg.dataset, alpha=float(ov["alpha"]))
    if "k_epochs" in ov:
        cfg.characterization = dataclasses.replace(cfg.characterization, k_epochs=int(ov["k_epochs"]))
    if "pop_size" in ov:
        cfg.evolution = dataclasses.replace(cfg.evolution, pop_size=int(ov["pop_size"]))
    if "n_generations" in ov:
        cfg.evolution = dataclasses.replace(cfg.evolution, n_generations=int(ov["n_generations"]))

    out = Path(args["output"])
    # skip if already completed
    if (out / "summary.json").exists() and not args.get("force", False):
        return {"status": "skipped", "output": str(out), "seed": seed, "config": ov, "pairing": args.get("pairing_mode")}

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)

    try:
        res = run_similarity_experiment(
            cfg,
            output_dir=out,
            verbose=False,
            pairing_mode=args.get("pairing_mode", "deterministic"),
            pairing_tau=args.get("pairing_tau", None),
            rep_mode=args.get("rep_mode", "alternating"),
            full_mode=args.get("full_mode", False),
        )
        return {"status": "completed", "output": str(out), "seed": seed, "pairing": args.get("pairing_mode"), "summary": res["summary"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        # save error
        try:
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "error.log", "w") as f:
                f.write(str(e) + "\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        return {"status": "failed", "output": str(out), "seed": seed, "error": str(e)}

def build_tasks():
    tasks = []
    # Experiment A: deterministic similarity baseline, 10 seeds
    for seed in SEEDS_10:
        tasks.append({
            "base_cfg_path": BASE_CFG_PATH,
            "output": str(BASE_OUTPUT / "A_deterministic_similarity" / f"seed_{seed:03d}"),
            "seed": seed,
            "overrides": {},
            "pairing_mode": "deterministic",
            "rep_mode": "alternating",
            "full_mode": False,
            "exp": "A",
        })
    # Experiment B: random pairing, 10 seeds
    for seed in SEEDS_10:
        tasks.append({
            "base_cfg_path": BASE_CFG_PATH,
            "output": str(BASE_OUTPUT / "B_random_pairing" / f"seed_{seed:03d}"),
            "seed": seed,
            "overrides": {},
            "pairing_mode": "random",
            "rep_mode": "alternating",
            "exp": "B",
        })
    # Experiment C: probabilistic tau sweep, 5 seeds each (could be 3 for speed, but use 5)
    for tau in TAU_VALUES:
        tau_str = f"{tau:.2f}".replace(".", "_")
        for seed in SEEDS_5:
            tasks.append({
                "base_cfg_path": BASE_CFG_PATH,
                "output": str(BASE_OUTPUT / "C_probabilistic_tau" / f"tau_{tau_str}" / f"seed_{seed:03d}"),
                "seed": seed,
                "overrides": {},
                "pairing_mode": "probabilistic",
                "pairing_tau": float(tau),
                "rep_mode": "alternating",
                "exp": f"C_tau_{tau}",
            })
    # Experiment D: N scaling, 5 seeds each
    for n in N_VALUES:
        for seed in SEEDS_5:
            tasks.append({
                "base_cfg_path": BASE_CFG_PATH,
                "output": str(BASE_OUTPUT / "D_node_scaling" / f"n_{n}" / f"seed_{seed:03d}"),
                "seed": seed,
                "overrides": {"n_clients": n},
                "pairing_mode": "deterministic",
                "rep_mode": "alternating",
                "exp": f"D_n{n}",
            })
    # Experiment E: heterogeneity alpha
    for alpha in ALPHA_VALUES:
        a_str = str(alpha).replace(".", "_")
        for seed in SEEDS_5:
            tasks.append({
                "base_cfg_path": BASE_CFG_PATH,
                "output": str(BASE_OUTPUT / f"E_heterogeneity" / f"alpha_{a_str}" / f"seed_{seed:03d}"),
                "seed": seed,
                "overrides": {"alpha": float(alpha)},
                "pairing_mode": "deterministic",
                "rep_mode": "alternating",
                "exp": f"E_alpha_{alpha}",
            })
    # Experiment F: K epochs
    for k in K_VALUES:
        for seed in SEEDS_5:
            tasks.append({
                "base_cfg_path": BASE_CFG_PATH,
                "output": str(BASE_OUTPUT / f"F_exploratory_K" / f"K_{k}" / f"seed_{seed:03d}"),
                "seed": seed,
                "overrides": {"k_epochs": int(k)},
                "pairing_mode": "deterministic",
                "rep_mode": "alternating",
                "exp": f"F_K{k}",
            })
    # Experiment G: rep strategy alternating vs random (both deterministic pairing)
    for rep in ["alternating", "random"]:
        for seed in SEEDS_5:
            tasks.append({
                "base_cfg_path": BASE_CFG_PATH,
                "output": str(BASE_OUTPUT / f"G_rep_strategy" / f"rep_{rep}" / f"seed_{seed:03d}"),
                "seed": seed,
                "overrides": {},
                "pairing_mode": "deterministic",
                "rep_mode": rep,
                "exp": f"G_{rep}",
            })
    # Experiment H: full evaluation baseline, 10 seeds
    for seed in SEEDS_10:
        tasks.append({
            "base_cfg_path": BASE_CFG_PATH,
            "output": str(BASE_OUTPUT / f"H_full_baseline" / f"seed_{seed:03d}"),
            "seed": seed,
            "overrides": {},
            "pairing_mode": "deterministic",
            "rep_mode": "alternating",
            "full_mode": True,
            "exp": "H",
        })
    return tasks

def run_baseline_previous():
    """Run previous algorithm baseline (I) using original runner if possible."""
    # This is optional but attempt to run 3 seeds of previous algorithm with same base config
    print("\n=== Experiment I: previous algorithm baseline (dissimilarity + NSGA-II) ===")
    outputs = []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from evofederated.experiments.runner import run_experiment as run_old
        from evofederated.utils.config import load_config
        import dataclasses
        base = load_config(BASE_CFG_PATH)
        # Adjust to match previous algorithm expectations: need to set baselines
        # We'll run 3 seeds
        for seed in [42,123,999]:
            cfg = dataclasses.replace(base)
            cfg.seed = seed
            cfg.dataset = dataclasses.replace(cfg.dataset, seed=seed, n_clients=8)
            cfg.characterization = dataclasses.replace(cfg.characterization, seed=seed)
            cfg.evolution = dataclasses.replace(cfg.evolution, seed=seed, pop_size=10, n_generations=8, strategy="dynamic")
            cfg.baselines = ("dynamic",)  # only dynamic dissimilarity
            out = BASE_OUTPUT / "I_baseline_previous" / f"seed_{seed:03d}"
            if (out / "summary.csv").exists():
                print(f"[SKIP I] seed {seed} exists")
                continue
            if out.exists():
                shutil.rmtree(out)
            try:
                res = run_old(cfg, output_dir=out, verbose=False)
                outputs.append(str(out))
                print(f"[I] completed seed {seed} -> {out}")
            except Exception as e:
                print(f"[I] failed seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"[I] baseline previous not available: {e}")
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="parallel workers (spawn)")
    parser.add_argument("--dry-run", action="store_true", help="print tasks without running")
    parser.add_argument("--skip-previous", action="store_true", help="skip I baseline")
    parser.add_argument("--only", type=str, default=None, help="only run specific exp group: A,B,C,D,E,F,G,H")
    args = parser.parse_args()

    tasks = build_tasks()
    # filter only if requested
    if args.only:
        allowed = set(args.only.split(","))
        # map letters to exp prefix
        # exp field like A, B, C_tau..., D_n..., etc. We'll filter by first char
        filtered = []
        for t in tasks:
            exp_letter = t["exp"][0]
            if exp_letter in allowed:
                filtered.append(t)
        tasks = filtered
        print(f"Filtered to {len(tasks)} tasks for {allowed}")

    print(f"Total tasks: {len(tasks)}")
    # Group counts
    from collections import Counter
    cnt = Counter([t["exp"] for t in tasks])
    for k,v in sorted(cnt.items())[:10]:
        print(f"  {k}: {v}")
    if args.dry_run:
        for t in tasks[:5]:
            print(t)
        return

    # Run baseline I first (sequential, small)
    if not args.skip_previous:
        run_baseline_previous()

    # Parallel execution
    start = time.time()
    workers = args.workers
    # Use spawn context to avoid torch fork issues
    ctx = multiprocessing.get_context("spawn")
    completed = 0
    failed = 0
    skipped = 0
    results = []
    # Limit threads per worker already handled in run_single_task
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {executor.submit(run_single_task, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                if res["status"] == "completed":
                    completed += 1
                elif res["status"] == "skipped":
                    skipped += 1
                else:
                    failed += 1
                print(f"[{res['status']}] {res.get('output','')} ({completed+skipped+failed}/{len(tasks)})")
            except Exception as e:
                failed += 1
                print(f"[exception] {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start
    print(f"\nPhase 2 execution finished in {elapsed:.1f}s")
    print(f"Completed: {completed}, Skipped: {skipped}, Failed: {failed}, Total: {len(tasks)}")
    # Save manifest
    manifest_path = BASE_OUTPUT / "phase2_manifest.json"
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"tasks": len(tasks), "completed": completed, "skipped": skipped, "failed": failed, "elapsed": elapsed, "results": results[:10]}, f, indent=2, default=str)
    print(f"Manifest at {manifest_path}")

if __name__ == "__main__":
    main()
