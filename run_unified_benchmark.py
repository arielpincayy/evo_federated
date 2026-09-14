#!/usr/bin/env python3
"""
Unified EvoFederated Benchmark - Compendio Experimental
Reconstruye y unifica las tres familias bajo infraestructura común.

Familias:
  A: Dissimilarity + NSGA-II (per pair, 2 objectives)
  B: Similarity + Single-Objective GA (representatives, mean accuracy)
  C: Many-Objective + NSGA-III / MOEA/D (per_silo and grouped)

Estructura results/unified_evofed_final/
  00_sanity/
  01_dissimilarity_nsga2/
  02_similarity_singleobjective/
  03_manyobjective/
  04_node_scaling/
  05_objective_reduction/
  06_heterogeneity/
  07_local_epochs/
  08_tau/
  09_representatives/
  10_full_baseline/
  11_final_comparison/
  report/

Cada run genera:
  config.json, runtime.json, generation_metrics.csv, individual_node_accuracy.csv,
  node_generation_summary.csv, population_metrics.csv, objective_vectors.csv,
  pairing.json, similarity_matrix.csv, distance_matrix.csv, final_full_evaluation.csv,
  pareto_front.csv (when applicable)

Trazabilidad:
  Similarity Single-Objective reconstructed from branch: experiment/similarity-pairing-single-objective (commit 43e8fcf + similarity_evaluator.py, matching.py, runner_similarity.py)
  Many-Objective NSGA-III/MOEA-D reconstructed from branch: experiment/many-objective-pca (untracked files many_objective.py, moead_runner.py, nsga3_runner.py, runner_many.py, grouping, PCA IGNORED per 0.6)
  Dissimilarity NSGA-II reconstructed from: main branch runner.py/evaluator.py/nsga2.py (divergence-based partial evaluation, reimplemented as per-pair global max-dissimilarity matching with 2 objectives, correction: old code used per-individual sampling, new uses per-pair independent optimization)

Uso:
  python run_unified_benchmark.py --sanity
  python run_unified_benchmark.py --full
  python run_unified_benchmark.py --only 01,02,03 --workers 8
  python run_unified_benchmark.py --dry-run
"""
import argparse, json, sys, dataclasses, shutil, time, platform
from pathlib import Path
import multiprocessing as mp
import concurrent.futures
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evofederated.utils.config import load_config, ExperimentConfig

BASE_RESULT = Path("results/unified_evofed_final")
CONFIG_SIM = "configs/similarity.yaml"
CONFIG_MANY = "configs/many_objective.yaml"
CONFIG_MAIN = "configs/small.yaml"  # for dissimilarity base (small.yaml is generic)

# Seeds per spec 39: 5 for exploration/sanity, 10 for central, 20 for final. Workstation allows substantial.
SEEDS_SANITY = [42, 123]
SEEDS_5 = [42, 123, 999, 2024, 2025]
SEEDS_10 = [42, 123, 999, 2024, 2025, 7, 11, 13, 17, 19]
SEEDS_3 = [42, 123, 999]

TAU_VALUES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 2.00, 5.00]
N_VALUES = [4, 8, 16, 32]
ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0, 10.0]
K_VALUES = [1, 2, 5, 10]
REP_STRATEGIES = ["alternating", "random", "probabilistic"]

def load_base_config(path):
    cfg=load_config(path)
    return cfg

def clone_cfg(base: ExperimentConfig, seed: int, overrides: dict = None):
    cfg = dataclasses.replace(base)
    cfg.seed = int(seed)
    cfg.dataset = dataclasses.replace(cfg.dataset, seed=int(seed))
    cfg.characterization = dataclasses.replace(cfg.characterization, seed=int(seed))
    cfg.evolution = dataclasses.replace(cfg.evolution, seed=int(seed))
    # Force CPU to avoid CUDA driver mismatch (12040) on this workstation
    cfg.train = dataclasses.replace(cfg.train, device="cpu")
    if overrides:
        if "n_clients" in overrides:
            cfg.dataset = dataclasses.replace(cfg.dataset, n_clients=int(overrides["n_clients"]))
        if "alpha" in overrides:
            cfg.dataset = dataclasses.replace(cfg.dataset, alpha=float(overrides["alpha"]))
        if "k_epochs" in overrides:
            cfg.characterization = dataclasses.replace(cfg.characterization, k_epochs=int(overrides["k_epochs"]))
        if "pop_size" in overrides:
            cfg.evolution = dataclasses.replace(cfg.evolution, pop_size=int(overrides["pop_size"]))
        if "n_generations" in overrides:
            cfg.evolution = dataclasses.replace(cfg.evolution, n_generations=int(overrides["n_generations"]))
        if "dataset_name" in overrides:
            cfg.dataset = dataclasses.replace(cfg.dataset, name=overrides["dataset_name"])
    return cfg

def worker_task(args):
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    task_type=args["task_type"]
    output=Path(args["output"])
    seed=args["seed"]
    # checkpoint/resume: if summary.json exists, skip
    if (output/"summary.json").exists() and not args.get("force",False):
        return {"status":"skipped","output":str(output),"task_type":task_type,"seed":seed}
    if (output/"config.json").exists() and (output/"individual_node_accuracy.csv").exists() and not args.get("force",False):
        return {"status":"skipped","output":str(output),"task_type":task_type,"seed":seed}
    if output.exists() and not (output/"summary.json").exists():
        # incomplete run, clean
        shutil.rmtree(output, ignore_errors=True)
    try:
        if task_type=="dissimilarity":
            from evofederated.experiments.runner_dissimilarity import run_dissimilarity_experiment
            from evofederated.utils.config import load_config
            import dataclasses
            base_cfg=load_config(args["base_cfg_path"])
            cfg=clone_cfg(base_cfg, seed, args.get("overrides"))
            res=run_dissimilarity_experiment(cfg, output_dir=output, verbose=False,
                                             pairing_mode=args.get("pairing_mode","deterministic"),
                                             pairing_tau=args.get("pairing_tau",None))
            return {"status":"completed","output":str(output),"task_type":task_type,"seed":seed}
        elif task_type=="similarity":
            from evofederated.experiments.runner_similarity import run_similarity_experiment
            from evofederated.utils.config import load_config
            import dataclasses
            base_cfg=load_config(args["base_cfg_path"])
            cfg=clone_cfg(base_cfg, seed, args.get("overrides"))
            res=run_similarity_experiment(cfg, output_dir=output, verbose=False,
                                          pairing_mode=args.get("pairing_mode","deterministic"),
                                          pairing_tau=args.get("pairing_tau",None),
                                          rep_mode=args.get("rep_mode","alternating"),
                                          full_mode=args.get("full_mode",False))
            return {"status":"completed","output":str(output),"task_type":task_type,"seed":seed}
        elif task_type=="many":
            from evofederated.experiments.runner_many_unified import run_many_unified_experiment
            from evofederated.utils.config import load_config
            import dataclasses
            base_cfg=load_config(args["base_cfg_path"])
            cfg=clone_cfg(base_cfg, seed, args.get("overrides"))
            res=run_many_unified_experiment(cfg, output_dir=output, verbose=False,
                                            optimizer=args.get("optimizer","nsga3"),
                                            objective_mode=args.get("objective_mode","per_silo"),
                                            grouping_strategy=args.get("grouping_strategy","similarity"),
                                            n_groups=args.get("n_groups",None),
                                            aggregation=args.get("aggregation","mean"))
            return {"status":"completed","output":str(output),"task_type":task_type,"seed":seed}
        else:
            return {"status":"failed","output":str(output),"error":f"unknown task_type {task_type}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            output.mkdir(parents=True, exist_ok=True)
            with open(output/"error.log","w") as f:
                f.write(str(e)+"\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        return {"status":"failed","output":str(output),"seed":seed,"error":str(e),"task_type":task_type}

def build_tasks(mode="sanity"):
    tasks=[]
    # Determine seeds/pop/gen based on mode
    if mode=="sanity":
        seeds=SEEDS_SANITY
        pop_small=5
        gen_small=3
        n_clients=4
    elif mode=="reduced":
        seeds=SEEDS_5
        pop_small=10
        gen_small=5
        n_clients=8
    else: # full
        seeds=SEEDS_10
        pop_small=10
        gen_small=8
        n_clients=8

    # 00_sanity: all three families with minimal config
    for family, ttype in [("dissimilarity","dissimilarity"),("similarity","similarity"),("many_nsga3","many")]:
        for seed in SEEDS_SANITY:
            if ttype=="dissimilarity":
                tasks.append({"task_type":"dissimilarity","base_cfg_path":CONFIG_MAIN,"output":str(BASE_RESULT/"00_sanity"/f"{family}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":4,"pop_size":5,"n_generations":3},"pairing_mode":"deterministic","exp":"00"})
            elif ttype=="similarity":
                tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"00_sanity"/f"{family}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":4,"pop_size":5,"n_generations":3},"pairing_mode":"deterministic","rep_mode":"alternating","exp":"00"})
            else:
                tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"00_sanity"/f"{family}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":4,"pop_size":5,"n_generations":3},"optimizer":"nsga3","objective_mode":"per_silo","exp":"00"})

    # 01_dissimilarity_nsga2: deterministic, random, probabilistic tau=0.5 as representative
    for pairing_mode, tau in [("deterministic",None),("random",None),("probabilistic",0.5)]:
        for seed in (SEEDS_5 if mode!="sanity" else SEEDS_SANITY):
            tau_str = f"_tau{str(tau).replace('.','_')}" if tau else ""
            tasks.append({"task_type":"dissimilarity","base_cfg_path":CONFIG_MAIN,"output":str(BASE_RESULT/"01_dissimilarity_nsga2"/f"{pairing_mode}{tau_str}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":pairing_mode,"pairing_tau":tau,"exp":"01"})

    # 02_similarity_singleobjective: deterministic, random, probabilistic tau0.5, full baseline; with alternating and random reps
    for pairing_mode, tau, rep in [("deterministic",None,"alternating"),("random",None,"alternating"),("probabilistic",0.5,"alternating"),("deterministic",None,"random")]:
        for seed in (SEEDS_5 if mode!="sanity" else SEEDS_SANITY):
            tau_str = f"_tau{str(tau).replace('.','_')}" if tau else ""
            tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"02_similarity_singleobjective"/f"{pairing_mode}{tau_str}_rep_{rep}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":pairing_mode,"pairing_tau":tau,"rep_mode":rep,"full_mode":False,"exp":"02"})
    # full baseline
    for seed in (SEEDS_5 if mode!="sanity" else SEEDS_SANITY):
        tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"02_similarity_singleobjective"/f"full_baseline"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","rep_mode":"alternating","full_mode":True,"exp":"02"})

    # 03_manyobjective: NSGA-III vs MOEA/D per_silo
    for optimizer in ["nsga3","moead"]:
        for seed in (SEEDS_5 if mode!="sanity" else SEEDS_SANITY):
            tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"03_manyobjective"/f"{optimizer}_per_silo"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"optimizer":optimizer,"objective_mode":"per_silo","exp":"03"})

    # 04_node_scaling: N=4,8,16,32 for each family representative
    for n in ([4,8] if mode=="sanity" else N_VALUES):
        # dissimilarity deterministic
        for seed in (SEEDS_SANITY if mode=="sanity" else SEEDS_3):
            tasks.append({"task_type":"dissimilarity","base_cfg_path":CONFIG_MAIN,"output":str(BASE_RESULT/"04_node_scaling"/f"n_{n}"/"dissimilarity"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":n,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","exp":"04"})
            tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"04_node_scaling"/f"n_{n}"/"similarity"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":n,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","rep_mode":"alternating","exp":"04"})
            tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"04_node_scaling"/f"n_{n}"/"nsga3_per_silo"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":n,"pop_size":pop_small,"n_generations":gen_small},"optimizer":"nsga3","objective_mode":"per_silo","exp":"04"})
            tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"04_node_scaling"/f"n_{n}"/"moead_per_silo"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":n,"pop_size":pop_small,"n_generations":gen_small},"optimizer":"moead","objective_mode":"per_silo","exp":"04"})

    # 05_objective_reduction: per_silo vs grouped similarity/dissimilarity/random mean/min, N/2 and N/4
    if mode!="sanity":
        for grouping_strategy in ["similarity","dissimilarity","random"]:
            for aggregation in ["mean","min"]:
                for n_groups in [4,2]:  # for N=8, N/2=4 groups size2, N/4=2 groups size4
                    for optimizer in ["nsga3","moead"]:
                        for seed in SEEDS_3:
                            tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"05_objective_reduction"/f"{grouping_strategy}_{aggregation}_g{n_groups}"/f"{optimizer}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"optimizer":optimizer,"objective_mode":"grouped","grouping_strategy":grouping_strategy,"n_groups":n_groups,"aggregation":aggregation,"exp":"05"})
        # per_silo baselines already in 03 but also add here as reference
        for optimizer in ["nsga3","moead"]:
            for seed in SEEDS_3:
                tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"05_objective_reduction"/f"per_silo"/f"{optimizer}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"optimizer":optimizer,"objective_mode":"per_silo","exp":"05"})

    # 06_heterogeneity: alpha 0.1,0.5,10.0 (subset for sanity) else full
    alphas = [0.1,0.5,10.0] if mode=="sanity" else ALPHA_VALUES
    for alpha in alphas:
        a_str=str(alpha).replace(".","_")
        for seed in (SEEDS_SANITY if mode=="sanity" else SEEDS_3):
            tasks.append({"task_type":"dissimilarity","base_cfg_path":CONFIG_MAIN,"output":str(BASE_RESULT/"06_heterogeneity"/f"alpha_{a_str}"/"dissimilarity"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"alpha":alpha,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","exp":"06"})
            tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"06_heterogeneity"/f"alpha_{a_str}"/"similarity"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"alpha":alpha,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","rep_mode":"alternating","exp":"06"})
            tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"06_heterogeneity"/f"alpha_{a_str}"/"nsga3"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"alpha":alpha,"pop_size":pop_small,"n_generations":gen_small},"optimizer":"nsga3","objective_mode":"per_silo","exp":"06"})

    # 07_local_epochs: K 1,2,5 (sanity: 1,2 else full)
    k_vals = [1,2] if mode=="sanity" else [1,2,5]
    if mode!="sanity":
        k_vals = K_VALUES
        # for sanity reduced, for reduced full, for full use all
        if mode=="reduced":
            k_vals=[1,2,5]
    for k in k_vals:
        for seed in (SEEDS_SANITY if mode=="sanity" else SEEDS_3):
            tasks.append({"task_type":"dissimilarity","base_cfg_path":CONFIG_MAIN,"output":str(BASE_RESULT/"07_local_epochs"/f"K_{k}"/"dissimilarity"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"k_epochs":k,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","exp":"07"})
            tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"07_local_epochs"/f"K_{k}"/"similarity"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"k_epochs":k,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","rep_mode":"alternating","exp":"07"})
            tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"07_local_epochs"/f"K_{k}"/"nsga3"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"k_epochs":k,"pop_size":pop_small,"n_generations":gen_small},"optimizer":"nsga3","objective_mode":"per_silo","exp":"07"})

    # 08_tau: tau sweep for both similarity and dissimilarity probabilistic
    taus = [0.1,0.5,1.0,2.0] if mode=="sanity" else [0.1,0.5,1.0,2.0,5.0]
    if mode!="sanity":
        taus = TAU_VALUES if mode=="full" else [0.1,0.5,1.0,2.0,5.0]
    for tau in taus:
        for seed in (SEEDS_SANITY if mode=="sanity" else SEEDS_3):
            tasks.append({"task_type":"dissimilarity","base_cfg_path":CONFIG_MAIN,"output":str(BASE_RESULT/"08_tau"/f"dissimilarity_tau_{str(tau).replace('.','_')}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"probabilistic","pairing_tau":tau,"exp":"08"})
            tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"08_tau"/f"similarity_tau_{str(tau).replace('.','_')}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"probabilistic","pairing_tau":tau,"rep_mode":"alternating","exp":"08"})

    # 09_representatives: alternating vs random vs probabilistic for similarity
    for rep in (["alternating","random"] if mode=="sanity" else REP_STRATEGIES):
        for seed in (SEEDS_SANITY if mode=="sanity" else SEEDS_3):
            tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"09_representatives"/f"rep_{rep}"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","rep_mode":rep,"exp":"09"})

    # 10_full_baseline: already in 02 but also separate folder for comprehensive full eval across families
    for seed in (SEEDS_SANITY if mode=="sanity" else SEEDS_5):
        tasks.append({"task_type":"dissimilarity","base_cfg_path":CONFIG_MAIN,"output":str(BASE_RESULT/"10_full_baseline"/"dissimilarity_random"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"random","exp":"10"})
        tasks.append({"task_type":"similarity","base_cfg_path":CONFIG_SIM,"output":str(BASE_RESULT/"10_full_baseline"/"similarity_full"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"pairing_mode":"deterministic","rep_mode":"alternating","full_mode":True,"exp":"10"})
        tasks.append({"task_type":"many","base_cfg_path":CONFIG_MANY,"output":str(BASE_RESULT/"10_full_baseline"/"many_nsga3_per_silo"/f"seed_{seed:03d}"),"seed":seed,"overrides":{"n_clients":8,"pop_size":pop_small,"n_generations":gen_small},"optimizer":"nsga3","objective_mode":"per_silo","exp":"10"})

    return tasks

def main():
    parser=argparse.ArgumentParser(description="Unified EvoFederated Benchmark")
    parser.add_argument("--sanity",action="store_true",help="run sanity small (N=4 pop5 gen3 2 seeds)")
    parser.add_argument("--reduced",action="store_true",help="run reduced benchmark (pop10 gen5 5 seeds)")
    parser.add_argument("--full",action="store_true",help="run full benchmark (pop10 gen8 10 seeds, may be heavy)")
    parser.add_argument("--workers",type=int,default=4,help="parallel workers")
    parser.add_argument("--dry-run",action="store_true",help="print tasks without executing")
    parser.add_argument("--only",type=str,default=None,help="only specific exp folders: 00,01 etc comma separated")
    parser.add_argument("--force",action="store_true",help="force rerun even if exists")
    args=parser.parse_args()
    if args.sanity:
        mode="sanity"
    elif args.reduced:
        mode="reduced"
    elif args.full:
        mode="full"
    else:
        mode="sanity"  # default
    tasks=build_tasks(mode=mode)
    if args.only:
        allowed=set(args.only.split(","))
        tasks=[t for t in tasks if t.get("exp") in allowed or any(t["output"].find(f"/{a}_")!=-1 for a in allowed)]
        print(f"Filtered to {len(tasks)} tasks for {allowed}")
    # also filter by force? no
    # group counts
    from collections import Counter
    cnt=Counter([t.get("exp","?") for t in tasks])
    print(f"Mode: {mode} | Total tasks: {len(tasks)}")
    for k,v in sorted(cnt.items()):
        print(f"  {k}: {v}")
    if args.dry_run:
        for t in tasks[:10]:
            print(t)
        print(f"... {len(tasks)} total")
        return
    # ensure base dirs
    BASE_RESULT.mkdir(parents=True, exist_ok=True)
    # save manifest
    with open(BASE_RESULT/"benchmark_manifest.json","w") as f:
        json.dump({"mode":mode,"total_tasks":len(tasks),"workers":args.workers,"tasks_sample":tasks[:5]},f,indent=2,default=str)
    start=time.time()
    workers=args.workers
    ctx=mp.get_context("spawn")
    completed=0; skipped=0; failed=0
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures={executor.submit(worker_task, {**t,"force":args.force}): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            try:
                res=fut.result()
                results.append(res)
                if res["status"]=="completed":
                    completed+=1
                elif res["status"]=="skipped":
                    skipped+=1
                else:
                    failed+=1
                print(f"[{res['status']}] {res.get('output','')} ({completed+skipped+failed}/{len(tasks)})")
            except Exception as e:
                failed+=1
                print(f"[exception] {e}")
                import traceback; traceback.print_exc()
    elapsed=time.time()-start
    print(f"\nBenchmark finished mode={mode} in {elapsed:.1f}s Completed:{completed} Skipped:{skipped} Failed:{failed} Total:{len(tasks)}")
    with open(BASE_RESULT/"benchmark_summary.json","w") as f:
        json.dump({"mode":mode,"elapsed":elapsed,"completed":completed,"skipped":skipped,"failed":failed,"total":len(tasks)},f,indent=2)
    # Also create aggregate and report
    print("Generating aggregate and report...")
    try:
        from generate_unified_report import generate_report
        generate_report(BASE_RESULT)
    except Exception as e:
        print(f"Report generation failed: {e}")
        import traceback; traceback.print_exc()

if __name__=="__main__":
    main()
