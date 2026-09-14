#!/usr/bin/env python3
"""Fase 2 batería many-objective: A-H con PCA factorial."""
import argparse
import json
import sys
from pathlib import Path
import concurrent.futures
import multiprocessing as mp
import time

sys.path.insert(0, "src")
from src.evofederated.utils.config import load_config
from src.evofederated.experiments.runner_many import run_many_objective_experiment
from src.evofederated.experiments.plots_many import generate_all_plots_many


def task_runner(task):
    # task: dict with keys
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    cfg_path = task["cfg_path"]
    out = Path(task["output_dir"])
    if (out / "summary.json").exists():
        return {"task": task, "status": "skipped", "out": str(out)}
    try:
        from src.evofederated.utils.config import load_config
        cfg = load_config(cfg_path)
        # override dataset/evolution params per task
        cfg.seed = int(task["seed"])
        cfg.dataset.seed = int(task["seed"])
        cfg.characterization.seed = int(task["seed"])
        cfg.evolution.seed = int(task["seed"])
        # dataset
        if "n_clients" in task:
            cfg.dataset.n_clients = int(task["n_clients"])
        if "alpha" in task:
            cfg.dataset.alpha = float(task["alpha"])
        if "k_epochs" in task:
            cfg.characterization.k_epochs = int(task["k_epochs"])
        # evolution pop/gen maybe overridden? keep default
        if "pop_size" in task:
            cfg.evolution.pop_size = int(task["pop_size"])
        if "n_generations" in task:
            cfg.evolution.n_generations = int(task["n_generations"])

        optimizer = task["optimizer"]
        pca_mode = task["pca_mode"]
        pca_variance = task.get("pca_variance", 0.95)
        pca_components = task.get("pca_components", 4)
        tau = task.get("tau", None)

        print(f"START {task['exp_id']} {optimizer}+{pca_mode} seed {task['seed']} N={cfg.dataset.n_clients} alpha={cfg.dataset.alpha} K={cfg.characterization.k_epochs} tau={tau} -> {out}")
        t0 = time.time()
        res = run_many_objective_experiment(
            cfg=cfg,
            optimizer=optimizer,
            pca_mode=pca_mode,
            pca_variance=pca_variance,
            pca_components=pca_components,
            output_dir=out,
            tau=tau,
            verbose=False,
        )
        generate_all_plots_many(out)
        elapsed = time.time() - t0
        print(f"DONE {task['exp_id']} seed {task['seed']} hv {res['hv_info']['hv_pareto']:.4f} time {elapsed:.1f}s")
        return {"task": task, "status": "done", "out": str(out), "hv": res['hv_info']['hv_pareto'], "time": elapsed}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"task": task, "status": f"fail: {e}", "out": str(out)}


def build_tasks(cfg_path, base_cfg, exp_root):
    exp_root = Path(exp_root)
    tasks = []
    # seeds
    SEEDS_10 = [42,123,999,2024,2025,7,11,13,17,19]
    SEEDS_5 = [42,123,999,2024,2025]
    TAU_LIST = [0.01,0.05,0.1,0.25,0.5,1.0,2.0,5.0]
    N_LIST = [4,8,16,32]
    ALPHA_LIST = [0.1,0.3,0.5,1.0,10.0]
    K_LIST = [1,2,5,10]
    PCA_VARIANCE_LIST = [0.90,0.95,0.99]
    PCA_FIXED_LIST = [2,4,8,16]

    base_path = Path(cfg_path)

    # A baseline principal: MOEAD/NSGA3 x no_pca / pca variance0.95, 10 seeds, N=8 alpha0.5 K2 (default)
    # Note phase1 already has 5 seeds for this config, we add remaining 5 to reach 10
    for opt in ["moead","nsga3"]:
        for pca_mode, pca_var in [("none",0.95),("variance",0.95)]:
            for seed in SEEDS_10:
                # check if already exists in phase1_validation to avoid duplicate but we want hierarchical new location
                # For phase2 baseline we use dedicated folder baseline/moead/no_pca etc.
                pca_dir = "no_pca" if pca_mode=="none" else "pca_variance0_95"
                out = exp_root / "baseline" / opt / pca_dir / f"seed_{seed:03d}"
                tasks.append({
                    "exp_id": "A_baseline",
                    "cfg_path": str(base_path),
                    "output_dir": str(out),
                    "optimizer": opt,
                    "pca_mode": pca_mode,
                    "pca_variance": pca_var,
                    "pca_components": 4,
                    "seed": seed,
                })

    # B tau sweep: optimizer x pca x tau, 5 seeds, N=8
    for opt in ["moead","nsga3"]:
        for pca_mode in ["none","variance"]:
            pca_var = 0.95 if pca_mode=="variance" else 0.95
            for tau in TAU_LIST:
                for seed in SEEDS_5:
                    pca_dir = "no_pca" if pca_mode=="none" else "pca"
                    out = exp_root / "tau" / f"{opt}_{pca_dir}_tau{str(tau).replace('.','_')}" / f"seed_{seed:03d}"
                    tasks.append({
                        "exp_id": "B_tau",
                        "cfg_path": str(base_path),
                        "output_dir": str(out),
                        "optimizer": opt,
                        "pca_mode": pca_mode,
                        "pca_variance": pca_var,
                        "tau": float(tau),
                        "seed": seed,
                    })

    # C node scaling: N 4,8,16,32 x optimizer x pca (none/variance), 5 seeds
    for n in N_LIST:
        for opt in ["moead","nsga3"]:
            for pca_mode in ["none","variance"]:
                pca_var = 0.95
                for seed in SEEDS_5:
                    pca_dir = "no_pca" if pca_mode=="none" else "pca"
                    out = exp_root / "node_scaling" / f"n{n}" / f"{opt}_{pca_dir}" / f"seed_{seed:03d}"
                    tasks.append({
                        "exp_id": f"C_n{n}",
                        "cfg_path": str(base_path),
                        "output_dir": str(out),
                        "optimizer": opt,
                        "pca_mode": pca_mode,
                        "pca_variance": pca_var,
                        "n_clients": int(n),
                        "seed": seed,
                    })

    # D heterogeneity alpha
    for alpha in ALPHA_LIST:
        for opt in ["moead","nsga3"]:
            for pca_mode in ["none","variance"]:
                for seed in SEEDS_5:
                    pca_dir = "no_pca" if pca_mode=="none" else "pca"
                    out = exp_root / "heterogeneity" / f"alpha{str(alpha).replace('.','_')}" / f"{opt}_{pca_dir}" / f"seed_{seed:03d}"
                    tasks.append({
                        "exp_id": f"D_alpha{alpha}",
                        "cfg_path": str(base_path),
                        "output_dir": str(out),
                        "optimizer": opt,
                        "pca_mode": pca_mode,
                        "pca_variance": 0.95,
                        "alpha": float(alpha),
                        "seed": seed,
                    })

    # E K epochs
    for k in K_LIST:
        for opt in ["moead","nsga3"]:
            for pca_mode in ["none","variance"]:
                for seed in SEEDS_5:
                    pca_dir = "no_pca" if pca_mode=="none" else "pca"
                    out = exp_root / "local_epochs" / f"K{k}" / f"{opt}_{pca_dir}" / f"seed_{seed:03d}"
                    tasks.append({
                        "exp_id": f"E_K{k}",
                        "cfg_path": str(base_path),
                        "output_dir": str(out),
                        "optimizer": opt,
                        "pca_mode": pca_mode,
                        "pca_variance": 0.95,
                        "k_epochs": int(k),
                        "seed": seed,
                    })

    # F PCA components: variance 0.90/95/99 and fixed 2/4/8/16, both optimizers, 5 seeds, N=8
    for pca_mode, val in [("variance",0.90),("variance",0.95),("variance",0.99)]:
        for opt in ["moead","nsga3"]:
            for seed in SEEDS_5:
                out = exp_root / "pca_components" / f"variance{str(val).replace('.','_')}" / f"{opt}" / f"seed_{seed:03d}"
                tasks.append({
                    "exp_id": f"F_pca_var{val}",
                    "cfg_path": str(base_path),
                    "output_dir": str(out),
                    "optimizer": opt,
                    "pca_mode": "variance",
                    "pca_variance": float(val),
                    "seed": seed,
                })
    for comp in PCA_FIXED_LIST:
        for opt in ["moead","nsga3"]:
            for seed in SEEDS_5:
                out = exp_root / "pca_components" / f"fixed{comp}" / f"{opt}" / f"seed_{seed:03d}"
                tasks.append({
                    "exp_id": f"F_pca_fixed{comp}",
                    "cfg_path": str(base_path),
                    "output_dir": str(out),
                    "optimizer": opt,
                    "pca_mode": "fixed",
                    "pca_components": int(comp),
                    "seed": seed,
                })

    # G similarity preservation is post-hoc, no extra runs (reuse F)
    # H full evaluation already baseline (full), no extra
    # I comparison reuse previous similarity results, no runs

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/many_objective.yaml")
    parser.add_argument("--output", default="results/many_objective_pca/phase2_experiments")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--filter", type=str, default=None, help="filter exp_id substring")
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    base_cfg = load_config(cfg_path)
    exp_root = Path(args.output)
    exp_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(cfg_path, base_cfg, exp_root)
    if args.filter:
        tasks = [t for t in tasks if args.filter in t["exp_id"]]
        print(f"Filtered to {len(tasks)} tasks containing '{args.filter}'")
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
        print(f"Limited to {len(tasks)} tasks")

    print(f"Total tasks: {len(tasks)}")
    # count by exp_id
    from collections import Counter
    cnt = Counter([t["exp_id"] for t in tasks])
    for k,v in cnt.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("Dry run, not executing")
        for t in tasks[:5]:
            print(t)
        return

    # check existing
    existing = sum(1 for t in tasks if (Path(t["output_dir"])/"summary.json").exists())
    print(f"Already completed: {existing}/{len(tasks)} (skipped)")
    print(f"To run: {len(tasks)-existing}")
    if existing == len(tasks):
        print("All done")
        return

    # execute
    ctx = mp.get_context("spawn")
    workers = args.workers
    print(f"Launching {len(tasks)} tasks with {workers} workers")
    start = time.time()
    results = []
    # Use chunks? Submit all
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as exe:
        futures = {exe.submit(task_runner, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                done = sum(1 for x in results if x["status"] in ["done","skipped"])
                fails = sum(1 for x in results if "fail" in x["status"])
                print(f"Progress {len(results)}/{len(tasks)} done+skip {done} fails {fails} | last {r['task']['exp_id']} seed {r['task']['seed']} status {r['status']}")
            except Exception as e:
                print(f"Future exception {e}")
                import traceback; traceback.print_exc()

    elapsed = time.time() - start
    print(f"All tasks finished in {elapsed:.1f}s")
    # save manifest
    manifest = {
        "total": len(tasks),
        "elapsed": elapsed,
        "results": results,
        "counts": dict(cnt),
    }
    with open(exp_root / "phase2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    # summary csv
    rows=[]
    for r in results:
        t=r["task"]
        status=r["status"]
        out=Path(t["output_dir"])
        summ_path=out/"summary.json"
        hv=0; mean_best=0; worst_best=0; D_red=0; pearson=0
        try:
            if summ_path.exists():
                s=json.load(open(summ_path))
                hv=s["hv"]["hv_pareto"]
                mean_best=s.get("mean_accuracy_best",0)
                worst_best=s.get("worst_accuracy_best",0)
                D_red=s.get("pca",{}).get("D_reduced",0)
                pearson=s.get("preservation",{}).get("pearson",0)
        except: pass
        rows.append({
            "exp_id": t["exp_id"],
            "optimizer": t["optimizer"],
            "pca_mode": t["pca_mode"],
            "seed": t["seed"],
            "n_clients": t.get("n_clients",8),
            "alpha": t.get("alpha",0.5),
            "k_epochs": t.get("k_epochs",2),
            "tau": t.get("tau",""),
            "status": status,
            "hv": hv,
            "mean_best": mean_best,
            "worst_best": worst_best,
            "D_red": D_red,
            "pearson": pearson,
            "out": str(out)
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(exp_root / "phase2_summary.csv", index=False)
    print(f"Summary saved to {exp_root / 'phase2_summary.csv'}")
    # quick stats
    df=pd.DataFrame(rows)
    print(df.groupby(["exp_id","optimizer","pca_mode"])["hv"].mean().head(20).to_string())


if __name__ == "__main__":
    main()
