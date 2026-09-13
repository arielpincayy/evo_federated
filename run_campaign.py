#!/usr/bin/env python3
"""Campaña experimental EvoFederated - battery with exhaustive validation, checkpoints, Hall of Fame."""
import time, json, sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evofederated.utils.config import load_config
from evofederated.experiments.runner import run_experiment

BASE = Path("configs/small.yaml")
OUT = Path("results/experimental_campaign")

def make_cfg(overrides=None):
    cfg = load_config(str(BASE))
    cfg.output_dir = str(OUT)
    cfg.seed = 42
    cfg.dataset.seed = 42
    cfg.characterization.seed = 42
    cfg.evolution.seed = 42
    cfg.dataset.name = "synthetic"
    cfg.dataset.alpha = 0.5
    cfg.dataset.n_clients = 6
    cfg.evolution.pop_size = 10
    cfg.evolution.n_generations = 5
    cfg.characterization.k_epochs = 2
    cfg.characterization.tau = 2.0
    cfg.characterization.distance_metric = "cosine"
    cfg.evolution.tau = 2.0
    cfg.baselines = ("full","random","fixed","dynamic")
    cfg.train.epochs = 2
    cfg.train.batch_size = 32
    cfg.train.device = "cpu"  # ponytail: force cpu for reproducibility + avoid cuda driver warns
    # checkpoint control: adapt proportionally
    # will be set per experiment based on n_generations
    cfg.evolution.control_every = 0
    cfg.evolution.control_top_k = 5
    if overrides:
        for k,v in overrides.items():
            parts=k.split(".")
            obj=cfg
            for p in parts[:-1]:
                obj=getattr(obj,p)
            setattr(obj, parts[-1], v)
        # auto set control_every if not explicitly set and generations provided
        if "evolution.n_generations" in overrides and "evolution.control_every" not in overrides:
            gen = cfg.evolution.n_generations
            if gen <= 3:
                cfg.evolution.control_every = 1
            elif gen <= 5:
                cfg.evolution.control_every = 2
            elif gen <= 10:
                cfg.evolution.control_every = 5
            else:
                cfg.evolution.control_every = 5
    else:
        # default for 5 gens
        cfg.evolution.control_every = 2
    return cfg

def run_one(name, cfg, subdir):
    out = OUT / subdir
    if (out/"summary.csv").exists():
        print(f"[SKIP] {name} already exists at {out}", flush=True)
        return out, True
    if out.exists() and not (out/"summary.csv").exists():
        print(f"[CLEAN] Removing incomplete {out}", flush=True)
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {name} -> {out} =====", flush=True)
    print(f" config: pop={cfg.evolution.pop_size} gen={cfg.evolution.n_generations} alpha={cfg.dataset.alpha} n_clients={cfg.dataset.n_clients} k={cfg.characterization.k_epochs} tau={cfg.evolution.tau} control_every={cfg.evolution.control_every}", flush=True)
    t0=time.time()
    try:
        res=run_experiment(cfg, output_dir=out, verbose=False)
        elapsed=time.time()-t0
        print(f"Done {name} in {elapsed:.1f}s. Summary:", flush=True)
        try:
            print(res["summary"].to_string(index=False), flush=True)
        except Exception:
            print(res["summary"], flush=True)
        with open(out/"campaign_meta.json","w") as f:
            json.dump({"name":name,"elapsed":elapsed,"subdir":str(subdir)},f,indent=2)
        return out, True
    except Exception as e:
        print(f"[FAIL] {name}: {e}", flush=True)
        import traceback; traceback.print_exc()
        with open(out/"FAILED.txt","w") as f:
            f.write(str(e))
        return out, False

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total_start=time.time()
    results_log=[]

    # Smoke
    cfg=make_cfg({"evolution.pop_size":4,"evolution.n_generations":2,"dataset.n_clients":4,"dataset.synthetic_n_samples":2000,"dataset.synthetic_n_features":20,"dataset.synthetic_n_classes":3,"evolution.control_every":1})
    out,ok=run_one("smoke-small",cfg,"smoke/small_pop4_gen2_c4")
    results_log.append(("smoke/small_pop4_gen2_c4",ok))
    cfg=make_cfg({"evolution.pop_size":6,"evolution.n_generations":3,"dataset.n_clients":6,"evolution.control_every":1})
    out,ok=run_one("smoke-baseline",cfg,"smoke/baseline_pop6_gen3")
    results_log.append(("smoke/baseline_pop6_gen3",ok))

    # Pop sensitivity
    for pop in [10,20,40]:
        cfg=make_cfg({"evolution.pop_size":pop,"evolution.n_generations":5})
        # pop40 is heavy but keep; if fails will log
        out,ok=run_one(f"pop_{pop}",cfg,f"population/pop_{pop}")
        results_log.append((f"population/pop_{pop}",ok))

    # Generations
    for gen in [5,10,20]:
        cfg=make_cfg({"evolution.pop_size":10,"evolution.n_generations":gen})
        out,ok=run_one(f"gen_{gen}",cfg,f"generations/gen_{gen}")
        results_log.append((f"generations/gen_{gen}",ok))

    # Alpha Non-IID
    for alpha in [10,1.0,0.5,0.1,0.05]:
        cfg=make_cfg({"dataset.alpha":alpha,"evolution.pop_size":10,"evolution.n_generations":5})
        name=f"alpha_{str(alpha).replace('.','_')}"
        # spec naming: alpha_10, alpha_1, alpha_05, alpha_01, alpha_005 – map 0.5->0_5 -> also create alias alpha_05 etc?
        # Keep our name, dashboard will parse alpha numeric.
        out,ok=run_one(name,cfg,f"alpha/{name}")
        results_log.append((f"alpha/{name}",ok))

    # Clients
    for nc in [5,10]:
        cfg=make_cfg({"dataset.n_clients":nc,"evolution.pop_size":10,"evolution.n_generations":5})
        out,ok=run_one(f"clients_{nc}",cfg,f"clients/clients_{nc}")
        results_log.append((f"clients/clients_{nc}",ok))
    # try 20 clients if feasible (keep but may be heavy)
    # We add 20 optionally, but skip if resource limited? Add with smaller pop to save
    try:
        cfg=make_cfg({"dataset.n_clients":20,"evolution.pop_size":10,"evolution.n_generations":5})
        out,ok=run_one("clients_20",cfg,"clients/clients_20")
        results_log.append(("clients/clients_20",ok))
    except Exception as e:
        print(f"clients_20 skipped: {e}")

    # k characterization
    for k in [1,3,5]:
        cfg=make_cfg({"characterization.k_epochs":k,"evolution.pop_size":10,"evolution.n_generations":5})
        out,ok=run_one(f"k_{k}",cfg,f"characterization_k/k_{k}")
        results_log.append((f"characterization_k/k_{k}",ok))

    # tau
    for tau in [0.5,1.0,2.0,5.0,10.0]:
        cfg=make_cfg({"characterization.tau":tau,"evolution.tau":tau,"evolution.pop_size":10,"evolution.n_generations":5})
        name=f"tau_{str(tau).replace('.','_')}"
        out,ok=run_one(name,cfg,f"tau/{name}")
        results_log.append((f"tau/{name}",ok))

    # Strategies comparison synthetic larger
    cfg=make_cfg({"evolution.pop_size":20,"evolution.n_generations":10,"dataset.n_clients":6,"dataset.alpha":0.5})
    out,ok=run_one("strat_synthetic_pop20_gen10",cfg,"strategy_comparison/synthetic_pop20_gen10")
    results_log.append(("strategy_comparison/synthetic_pop20_gen10",ok))
    # Also create alias under strategies/ for spec compliance
    try:
        alias = OUT / "strategies" / "synthetic_pop20_gen10"
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists() and (OUT/"strategy_comparison/synthetic_pop20_gen10").exists():
            # symlink or copy minimal – just create a marker file linking
            import os
            try:
                os.symlink(os.path.abspath(OUT/"strategy_comparison/synthetic_pop20_gen10"), alias)
            except Exception:
                pass
    except Exception:
        pass

    # FMNIST medium
    cfg_fmnist=load_config(str(BASE))
    cfg_fmnist.dataset.name="fashion-mnist"
    cfg_fmnist.dataset.data_root="./data"
    cfg_fmnist.dataset.n_clients=6
    cfg_fmnist.dataset.alpha=0.5
    cfg_fmnist.seed=42
    cfg_fmnist.dataset.seed=42
    cfg_fmnist.train.epochs=2
    cfg_fmnist.train.batch_size=64
    cfg_fmnist.train.device="cpu"
    cfg_fmnist.characterization.k_epochs=2
    cfg_fmnist.characterization.tau=2.0
    cfg_fmnist.characterization.distance_metric="cosine"
    cfg_fmnist.evolution.pop_size=10
    cfg_fmnist.evolution.n_generations=5
    cfg_fmnist.evolution.tau=2.0
    cfg_fmnist.evolution.control_every=2
    cfg_fmnist.evolution.control_top_k=5
    cfg_fmnist.baselines=("full","random","fixed","dynamic")
    cfg_fmnist.output_dir=str(OUT)
    cfg_fmnist.genome.l_max=3
    out,ok=run_one("fmnist_pop10_gen5",cfg_fmnist,"strategy_comparison/fashion_mnist_pop10_gen5")
    results_log.append(("strategy_comparison/fashion_mnist_pop10_gen5",ok))
    # alias for strategies
    try:
        alias = OUT / "strategies" / "fashion_mnist_pop10_gen5"
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists() and (OUT/"strategy_comparison/fashion_mnist_pop10_gen5").exists():
            import os
            try:
                os.symlink(os.path.abspath(OUT/"strategy_comparison/fashion_mnist_pop10_gen5"), alias)
            except Exception:
                pass
    except Exception:
        pass

    # Multi-seed baseline
    seeds=[42,123,999]
    for seed in seeds:
        cfg=make_cfg({"evolution.pop_size":10,"evolution.n_generations":5,"dataset.alpha":0.5})
        cfg.seed=seed
        cfg.dataset.seed=seed
        cfg.characterization.seed=seed
        cfg.evolution.seed=seed
        out,ok=run_one(f"multiseed_baseline_seed{seed}",cfg,f"multi_seed/baseline_alpha0_5/seed_{seed}")
        results_log.append((f"multi_seed/baseline_alpha0_5/seed_{seed}",ok))

    for seed in seeds:
        cfg=make_cfg({"evolution.pop_size":10,"evolution.n_generations":5,"dataset.alpha":0.05})
        cfg.seed=seed
        cfg.dataset.seed=seed
        cfg.characterization.seed=seed
        cfg.evolution.seed=seed
        out,ok=run_one(f"multiseed_alpha0.05_seed{seed}",cfg,f"multi_seed/alpha0_05/seed_{seed}")
        results_log.append((f"multi_seed/alpha0_05/seed_{seed}",ok))

    for seed in seeds[:2]:
        cfg=make_cfg({"evolution.pop_size":10,"evolution.n_generations":5,"dataset.alpha":0.1,"characterization.tau":5.0,"evolution.tau":5.0})
        cfg.seed=seed
        cfg.dataset.seed=seed
        cfg.characterization.seed=seed
        cfg.evolution.seed=seed
        out,ok=run_one(f"multiseed_alpha0_1_tau5_seed{seed}",cfg,f"multi_seed/alpha0_1_tau5/seed_{seed}")
        results_log.append((f"multi_seed/alpha0_1_tau5/seed_{seed}",ok))

    # Create aggregated directories for spec compliance
    for name in ["pareto_checkpoints", "hall_of_fame", "global_validation"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    # Populate pareto_checkpoints with symlinks/copies of checkpoint data
    try:
        import glob as _glob
        import pandas as pd
        # collect all checkpoint summaries
        ckpt_files = list((OUT).rglob("checkpoints/summary.csv"))
        if ckpt_files:
            rows=[]
            for f in ckpt_files:
                try:
                    df=pd.read_csv(f)
                    df["source"]=str(f.relative_to(OUT))
                    rows.append(df)
                except Exception:
                    pass
            if rows:
                pd.concat(rows).to_csv(OUT/"pareto_checkpoints"/"all_checkpoints_summary.csv", index=False)
        hof_files = list((OUT).rglob("hall_of_fame.csv"))
        if hof_files:
            rows=[]
            for f in hof_files:
                try:
                    df=pd.read_csv(f)
                    df["source"]=str(f.relative_to(OUT))
                    rows.append(df)
                except Exception:
                    pass
            if rows:
                pd.concat(rows, ignore_index=True).to_csv(OUT/"hall_of_fame"/"all_hall_of_fame.csv", index=False)
        exh_files = list((OUT).rglob("exhaustive_validation.csv"))
        # Use top-level exhaustive already? But also create global_validation aggregated
        if exh_files:
            rows=[]
            for f in exh_files:
                if "pareto_checkpoints" in str(f) or "hall_of_fame" in str(f):
                    continue
                try:
                    df=pd.read_csv(f)
                    # only those that are strategy-level (have strategy col)
                    if "mean_f1" in df.columns:
                        rows.append(df)
                except Exception:
                    pass
            if rows:
                # take top-level combined if exists else concat
                top = OUT/"exhaustive_validation.csv"
                if not top.exists():
                    pd.concat(rows).to_csv(OUT/"global_validation"/"global_exhaustive.csv", index=False)
                else:
                    import shutil
                    shutil.copy(top, OUT/"global_validation"/"global_exhaustive.csv")
    except Exception as e:
        print(f"aggregation for spec dirs failed: {e}")

    total_elapsed=time.time()-total_start
    print(f"\n=== Campaign finished in {total_elapsed/60:.1f} min ===")
    ok_count=sum(1 for _,ok in results_log if ok)
    fail_count=len(results_log)-ok_count
    print(f"Total experiments: {len(results_log)}, OK: {ok_count}, FAIL: {fail_count}")
    for name,ok in results_log:
        print(f" {'OK' if ok else 'FAIL'} {name}")
    with open(OUT/"campaign_summary.json","w") as f:
        json.dump({"total_elapsed_sec":total_elapsed,"total":len(results_log),"ok":ok_count,"fail":fail_count,"log":results_log},f,indent=2)

if __name__=="__main__":
    main()
