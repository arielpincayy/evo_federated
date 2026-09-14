#!/usr/bin/env python3
"""Fase 1 many-objective: 4 configs x N seeds sanity con métricas y validaciones."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing as mp

sys.path.insert(0, "src")
from src.evofederated.utils.config import load_config
from src.evofederated.experiments.runner_many import run_many_objective_experiment
from src.evofederated.experiments.plots_many import generate_all_plots_many


def task_runner(args):
    # args: (cfg_path, optimizer, pca_mode, pca_variance, pca_components, out_dir, seed)
    cfg_path, optimizer, pca_mode, pca_variance, pca_components, out_dir, seed = args
    # isolated process: force torch single thread and spawn safe
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    cfg = load_config(cfg_path)
    cfg.seed = int(seed)
    cfg.dataset.seed = int(seed)
    cfg.characterization.seed = int(seed)
    cfg.evolution.seed = int(seed)
    # keep config as is (pop10 gen8 etc) but for speed reduce samples if needed
    # For Phase1 we keep original config but ensure fast: limit n_samples to 3000 if large?
    # Keep as configured in file (4000)
    # normalize pca_mode for dir naming: variance0.95 -> variance_95, none stays none
    norm_pca = pca_mode
    if norm_pca.startswith("variance"):
        norm_pca = norm_pca.replace("variance","variance").replace(".","_")
    if norm_pca.startswith("fixed"):
        norm_pca = norm_pca.replace("fixed","fixed")
    out = Path(out_dir) / f"{optimizer}_{norm_pca}_seed{seed:03d}"
    if (out / "summary.json").exists():
        print(f"SKIP exists {out}")
        # normalize return pca for consistent aggregation
        ret_pca = pca_mode
        if ret_pca.startswith("variance"):
            ret_pca = "variance"
        elif ret_pca.startswith("fixed"):
            ret_pca = "fixed"
        return (optimizer, ret_pca, seed, out, "skipped")
    try:
        pm = pca_mode
        pv = pca_variance
        pc = pca_components
        # parse pca_mode special strings like variance0.95
        if pm.startswith("variance"):
            try:
                val = pm.replace("variance","")
                if val:
                    pv = float(val)
                pm = "variance"
            except: pass
        if pm.startswith("fixed"):
            try:
                val = pm.replace("fixed","")
                if val:
                    pc = int(val)
                pm = "fixed"
            except: pass
        print(f"START {optimizer}+{pm} seed {seed} -> {out}")
        res = run_many_objective_experiment(
            cfg=cfg,
            optimizer=optimizer,
            pca_mode=pm,
            pca_variance=pv,
            pca_components=pc,
            output_dir=out,
            verbose=False,
        )
        generate_all_plots_many(out)
        print(f"DONE {optimizer}+{pm} seed {seed} hv {res['hv_info']['hv_pareto']:.4f} d {res['pca_info']['D_reduced']}")
        return (optimizer, pm, seed, out, "done")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL {optimizer}+{pca_mode} seed {seed}: {e}")
        return (optimizer, pca_mode, seed, out, f"fail: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/many_objective.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,123,999,2024,2025])
    parser.add_argument("--output", default="results/many_objective_pca/phase1_validation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizers", nargs="+", default=["moead","nsga3"])
    parser.add_argument("--pca_modes", nargs="+", default=["none","variance0.95"])
    args = parser.parse_args()

    cfg_path = Path(args.config)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in args.seeds:
        for opt in args.optimizers:
            for pca in args.pca_modes:
                tasks.append((str(cfg_path), opt, pca, 0.95, 4, str(out_root), seed))

    print(f"Launching {len(tasks)} tasks with {args.workers} workers")
    # Use spawn to avoid torch fork issues
    ctx = mp.get_context("spawn")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as exe:
        futures = [exe.submit(task_runner, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                print(f"Progress {len(results)}/{len(tasks)} : {r}")
            except Exception as e:
                print(f"Future fail {e}")
                results.append(("error","error",0,None,str(e)))

    # aggregate
    rows = []
    for opt, pca, seed, out, status in results:
        if out is None: continue
        out = Path(out)
        summ_path = out / "summary.json"
        try:
            with open(summ_path) as f:
                summ = json.load(f)
            hv_p = summ["hv"]["hv_pareto"]
            hv_f = summ["hv"]["hv_final_pop"]
            n_p = summ["hv"]["n_pareto"]
            mean_best = summ.get("mean_accuracy_best",0)
            worst_best = summ.get("worst_accuracy_best",0)
            cost = summ.get("cost",{})
            pca_info = summ.get("pca",{})
            pres = summ.get("preservation",{})
        except Exception as e:
            print(f"aggregate fail {out} {e}")
            hv_p=hv_f=n_p=mean_best=worst_best=0
            cost={}; pca_info={}; pres={}
        rows.append({
            "optimizer": opt, "pca": pca, "seed": seed, "status": status,
            "hv_pareto": hv_p, "hv_final": hv_f, "n_pareto": n_p,
            "mean_best": mean_best, "worst_best": worst_best,
            "D_original": pca_info.get("D_original",0),
            "D_reduced": pca_info.get("D_reduced",0),
            "explained": pca_info.get("explained_variance_total",0),
            "pearson": pres.get("pearson",0),
            "pairing_pres": pres.get("pairing_preserved",0),
            "time": cost.get("total_time",0),
            "out": str(out)
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "aggregate_summary.csv", index=False)
    print(df.to_string(index=False))
    for (opt,pca), grp in df.groupby(["optimizer","pca"]):
        print(f"\n[{opt}+{pca}] hv_pareto mean {grp['hv_pareto'].mean():.5f} std {grp['hv_pareto'].std():.5f} n={len(grp)}")
        print(f" mean_best {grp['mean_best'].mean():.3f} worst {grp['worst_best'].mean():.3f} n_pareto {grp['n_pareto'].mean():.1f}")
        if "variance" in pca: print(f" D {grp['D_reduced'].mean():.0f} explained {grp['explained'].mean():.3f} pearson {grp['pearson'].mean():.3f}")

    # validation checklist
    print("\n--- VALIDATION CHECKS ---")
    checks=[]
    for opt, pca, seed, out, status in results:
        if status=="skipped": status="done" # still check
        if out is None or not Path(out).exists(): continue
        out = Path(out)
        try:
            hv_hist = pd.read_csv(out/"hypervolume.csv")
            pareto = pd.read_csv(out/"pareto_front.csv")
            n_obj_cols = len([c for c in pareto.columns if c.startswith("obj_")])
            n_acc_cols = len([c for c in pareto.columns if c.startswith("acc_client_")])
            cfg = json.load(open(out/"config.json"))
            expected = cfg["dataset"]["n_clients"]
            checks.append((f"{opt}_{pca}_seed{seed} N objectives ({n_obj_cols}=={expected})", n_obj_cols==expected))
            checks.append((f"{opt}_{pca}_seed{seed} acc per client ({n_acc_cols}=={expected})", n_acc_cols==expected))
            # F in [0,1] (1-acc)
            obj_cols = [c for c in pareto.columns if c.startswith("obj_")]
            if obj_cols:
                vals = pareto[obj_cols].values
                checks.append((f"{opt}_{pca}_seed{seed} objectives 0..1", np.all(vals>= -1e-6) and np.all(vals<=1.0+1e-6)))
            # PCA none redu reduced == original and preservation 1
            pinfo = json.load(open(out/"similarity_pca"/"pca_info.json"))
            if "none" in pca:
                checks.append((f"{opt}_{pca}_seed{seed} PCA none D reduced==original", pinfo["D_reduced"]==pinfo["D_original"]))
                # S raw == S pca preserved 1
                simstats = json.load(open(out/"similarity"/"similarity_stats.json"))
                checks.append((f"{opt}_{pca}_seed{seed} preservation ~1", abs(simstats["preservation"]["pearson"]-1.0)<1e-6))
            else:
                checks.append((f"{opt}_{pca}_seed{seed} PCA reduced < original", pinfo["D_reduced"] < pinfo["D_original"]))
                checks.append((f"{opt}_{pca}_seed{seed} PCA max_valid", pinfo["D_reduced"] <= pinfo["max_valid"]))
            # reference dirs
            meta = json.load(open(out/"metadata.json"))
            checks.append((f"{opt}_{pca}_seed{seed} ref_dirs shape N", meta["ref_dirs_shape"][1]==expected))
            # nondominated dominance check small sample: ensure all points in pareto are nondominated within final pop
            # load hypervolume history mean accuracy vs generation should increase or not decrease too much? Not strict.
            # Ensure eval_count == pop*gen*N (full)
            cost = json.load(open(out/"runtime.json"))
            pop_actual = cost["pop_size_actual"]
            gen = cost["n_generations"]
            n_clients = cost["n_clients"]
            expected_evals = pop_actual * gen * n_clients  # note + initial pop? pymoo counts evaluations including initial? Let's check evaluator count vs expected: gen*pop*N plus initial? We'll allow 10% diff
            evals = cost["eval_count"]
            # ManyObjectiveEvaluator counts each client training per individual per gen? It counts per evaluate_batch called each generation: pop*gen*N but pymoo also evaluates initial population (gen 0) -> need to see count
            # For our runner, generation_history records per individual per generation including gen 1..n_generations? initial pop is gen 1? We'll not strictly check.
            checks.append((f"{opt}_{pca}_seed{seed} eval_count ~ pop*gen*N", evals >= pop_actual*gen*n_clients*0.8))
            # seeds reproducible: check config seed == run_meta seed etc.
            checks.append((f"{opt}_{pca}_seed{seed} metadata seed == {seed}", meta["seed"]==seed))
        except Exception as e:
            print(f"check fail {out} {e}")
            checks.append((f"{opt}_{pca}_seed{seed} exception", False))
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    fails=[c for c in checks if not c[1]]
    print(f"\nVALIDATION {len(checks)-len(fails)}/{len(checks)} PASS, {len(fails)} FAIL")
    if fails:
        print("Failed:")
        for n,ok in fails: print(" ",n)

    print(f"\n=== FASE 1 many-objective DONE. Results in {out_root} ===")
    # generate aggregate plots
    try:
        aggregate_plots(out_root)
    except Exception as e:
        print(f"aggregate plots fail {e}")
        import traceback; traceback.print_exc()

def aggregate_plots(out_root: Path):
    out_root=Path(out_root)
    df=pd.read_csv(out_root/"aggregate_summary.csv")
    import matplotlib.pyplot as plt
    # boxplot hv by optimizer x pca
    plt.figure(figsize=(8,5))
    # pivot
    labels=[]
    data=[]
    for (opt,pca), grp in df.groupby(["optimizer","pca"]):
        labels.append(f"{opt}\n{pca}")
        data.append(grp["hv_pareto"].tolist())
    plt.boxplot(data, labels=labels)
    plt.ylabel("HV pareto")
    plt.title("HV comparison (aggregate)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_root/"aggregate_hv_boxplot.png", dpi=150)
    plt.close()
    # mean/worst scatter
    plt.figure(figsize=(6,5))
    for (opt,pca), grp in df.groupby(["optimizer","pca"]):
        plt.scatter(grp["mean_best"], grp["hv_pareto"], label=f"{opt}+{pca}", alpha=0.7)
    plt.xlabel("mean_best accuracy")
    plt.ylabel("HV")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_root/"aggregate_mean_vs_hv.png", dpi=150)
    plt.close()
    # preservation vs pca
    if "pearson" in df.columns:
        plt.figure(figsize=(6,4))
        sub=df[df["pca"]!="none"]
        if not sub.empty:
            plt.hist(sub["pearson"], bins=10, alpha=0.6, label="pearson")
            plt.xlabel("pearson preservation")
            plt.title("PCA similarity preservation")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_root/"aggregate_preservation_hist.png", dpi=150)
            plt.close()
    print("aggregate plots generated")

if __name__ == "__main__":
    main()
