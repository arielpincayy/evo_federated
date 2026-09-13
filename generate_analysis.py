#!/usr/bin/env python3
"""Genera aggregate_results.csv, experiment_index.csv, analysis.json y dashboard HTML.
Ponytail: simple, stdlib + pandas/numpy/matplotlib ya instalados.
"""
import json, csv, sys
from pathlib import Path
import pandas as pd
import numpy as np
import base64

ROOT = Path("results/experimental_campaign")
OUT_DASH = ROOT / "dashboard"
OUT_DASH.mkdir(parents=True, exist_ok=True)

# Recolectar experimentos: cada directorio que contiene summary.csv con config.json (experimento real)
experiments = []
for p in ROOT.rglob("summary.csv"):
    exp_dir = p.parent
    # only consider experiment dirs that have config.json and are not checkpoints
    if "checkpoints" in str(p):
        continue
    if not (exp_dir / "config.json").exists():
        continue
    rel = exp_dir.relative_to(ROOT)
    if str(rel) == ".":
        continue
    experiments.append(exp_dir)

# Ordenar por path
experiments = sorted(experiments)

print(f"Found {len(experiments)} experiments with summary.csv")

# Experiment index
index_rows = []
aggregate_rows = []

total_evals = 0
total_time = 0.0

for exp in experiments:
    rel = exp.relative_to(ROOT)
    # load config
    cfg_path = exp / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    # dataset params
    dataset = cfg.get("dataset", {})
    evol = cfg.get("evolution", {})
    char = cfg.get("characterization", {})
    train = cfg.get("train", {})
    # metadata
    meta_path = exp / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            try:
                meta = json.load(f)
            except:
                meta = {}
    elapsed = meta.get("elapsed_seconds", 0)
    total_time += float(elapsed) if elapsed else 0
    # summary
    df_summary = pd.read_csv(exp / "summary.csv")
    if "strategy" not in df_summary.columns:
        print(f"[SKIP] {rel} summary without strategy col")
        continue
    # proxy validation
    proxy_path = exp / "characterization" / "proxy_validation.json"
    proxy = {}
    if proxy_path.exists():
        with open(proxy_path) as f:
            proxy = json.load(f)
    # heterogeneity
    het_path = exp / "dataset" / "heterogeneity.json"
    het = {}
    if het_path.exists():
        with open(het_path) as f:
            try:
                data = json.load(f)
                het = data.get("heterogeneity", {})
            except:
                het = {}
    # campaign meta
    camp_meta_path = exp / "campaign_meta.json"
    camp_meta = {}
    if camp_meta_path.exists():
        with open(camp_meta_path) as f:
            camp_meta = json.load(f)

    # row for index
    index_rows.append({
        "experiment": str(rel),
        "dataset": dataset.get("name",""),
        "n_clients": dataset.get("n_clients",""),
        "alpha": dataset.get("alpha",""),
        "pop_size": evol.get("pop_size",""),
        "n_generations": evol.get("n_generations",""),
        "k_epochs": char.get("k_epochs",""),
        "tau": evol.get("tau",""),
        "distance_metric": char.get("distance_metric",""),
        "seed": cfg.get("seed",""),
        "elapsed_sec": elapsed,
        "proxy_spearman": proxy.get("spearman",""),
        "proxy_pearson": proxy.get("pearson",""),
        "mean_pairwise_TV": het.get("mean_pairwise_TV",""),
        "cv_size": het.get("cv_size",""),
    })

    # aggregate per strategy
    for _, row in df_summary.iterrows():
        strat = row["strategy"]
        hv_common = float(row.get("hv_common_validation",0))
        hv_internal = float(row.get("hv_final_internal",0))
        eval_count = int(row.get("eval_count",0))
        saving = float(row.get("saving_percent",0)) if "saving_percent" in row else 0
        total_evals += eval_count
        # exhaustive stats
        agg = {
            "experiment": str(rel),
            "strategy": strat,
            "hv_common": hv_common,
            "hv_internal": hv_internal,
            "eval_count": eval_count,
            "eval_count_evo_only": int(row.get("eval_count_evolution_only",0)),
            "char_cost": int(row.get("char_cost",0)),
            "saving_percent": saving,
            "mean_of_mean_f1": float(row.get("mean_of_mean_f1_exhaustive",0)),
            "median_of_mean_f1": float(row.get("median_of_mean_f1",0)),
            "best_mean_f1": float(row.get("best_mean_f1",0)),
            "mean_of_min_f1": float(row.get("mean_of_min_f1",0)),
            "worst_min_f1": float(row.get("worst_min_f1",0)),
            "dataset": dataset.get("name",""),
            "n_clients": dataset.get("n_clients",""),
            "alpha": dataset.get("alpha",""),
            "pop_size": evol.get("pop_size",""),
            "n_generations": evol.get("n_generations",""),
            "k_epochs": char.get("k_epochs",""),
            "tau": evol.get("tau",""),
            "seed": cfg.get("seed",""),
            "elapsed_sec": elapsed,
            "proxy_spearman": proxy.get("spearman",""),
            "proxy_pearson": proxy.get("pearson",""),
            "mean_pairwise_TV": het.get("mean_pairwise_TV",""),
            "hv_per_eval": hv_common/eval_count if eval_count else 0,
        }
        aggregate_rows.append(agg)

# save CSVs
df_index = pd.DataFrame(index_rows)
df_index.to_csv(ROOT / "experiment_index.csv", index=False)
print(f"Wrote experiment_index.csv {len(df_index)} rows")

df_agg = pd.DataFrame(aggregate_rows)
df_agg.to_csv(ROOT / "aggregate_results.csv", index=False)
print(f"Wrote aggregate_results.csv {len(df_agg)} rows, total_evals {total_evals}, total_time {total_time:.1f}s")

# Analysis JSON - compute key findings
# Helper to get group
def get_exp(rel):
    sub = df_agg[df_agg["experiment"]==rel]
    return sub

# Strategy comparison primary: synthetic_pop20_gen10 and fashion_mnist
def strategy_comparison(rel):
    sub = df_agg[df_agg["experiment"]==rel]
    if sub.empty:
        return {}
    # pivot
    res = {}
    for _, r in sub.iterrows():
        s=r["strategy"]
        res[s]= {
            "hv_common": r["hv_common"],
            "mean_f1": r["mean_of_mean_f1"],
            "worst_f1": r["worst_min_f1"],
            "eval_count": int(r["eval_count"]),
            "saving": r["saving_percent"],
            "hv_per_eval": r["hv_per_eval"]
        }
    # ratios dynamic/full
    if "full" in res and "dynamic" in res:
        res["ratio_dynamic_full_hv"] = res["dynamic"]["hv_common"]/res["full"]["hv_common"] if res["full"]["hv_common"] else 0
        res["ratio_dynamic_full_evals"] = res["dynamic"]["eval_count"]/res["full"]["eval_count"] if res["full"]["eval_count"] else 0
    return res

synth_comp = strategy_comparison("strategy_comparison/synthetic_pop20_gen10")
fmnist_comp = strategy_comparison("strategy_comparison/fashion_mnist_pop10_gen5")

# Alpha trend
alpha_rows = df_agg[df_agg["experiment"].str.startswith("alpha/")].copy()
# sort by alpha
alpha_rows["alpha_num"] = pd.to_numeric(alpha_rows["alpha"], errors="coerce")
alpha_pivot = alpha_rows.pivot_table(index="alpha_num", columns="strategy", values=["hv_common","mean_of_mean_f1"], aggfunc="mean")

# Pop trend
pop_rows = df_agg[df_agg["experiment"].str.startswith("population/")]
pop_pivot = pop_rows.pivot_table(index="pop_size", columns="strategy", values=["hv_common","mean_of_mean_f1"], aggfunc="mean")

# Gen trend
gen_rows = df_agg[df_agg["experiment"].str.startswith("generations/")]
gen_pivot = gen_rows.pivot_table(index="n_generations", columns="strategy", values="hv_common", aggfunc="mean")

# K trend
k_rows = df_agg[df_agg["experiment"].str.startswith("characterization_k/")]
k_pivot = k_rows.pivot_table(index="k_epochs", columns="strategy", values="hv_common", aggfunc="mean")

# Tau trend
tau_rows = df_agg[df_agg["experiment"].str.startswith("tau/")]
tau_pivot = tau_rows.pivot_table(index="tau", columns="strategy", values="hv_common", aggfunc="mean")

# Proxy correlations mean
proxy_vals = df_index["proxy_spearman"].dropna()
proxy_vals = pd.to_numeric(proxy_vals, errors="coerce").dropna()

# Multi-seed stats
ms_baseline = df_agg[df_agg["experiment"].str.contains("multi_seed/baseline")]
ms_alpha05 = df_agg[df_agg["experiment"].str.contains("alpha0_05")]
# compute mean/std per strategy
def ms_stats(df):
    out={}
    for strat, g in df.groupby("strategy"):
        vals = g["hv_common"].values
        out[strat]= {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals)>1 else 0), "n": int(len(vals)), "values": vals.tolist()}
    return out

ms_baseline_stats = ms_stats(ms_baseline)
ms_alpha05_stats = ms_stats(ms_alpha05)

# Build analysis dict
analysis = {
    "meta": {
        "total_experiments": len(index_rows),
        "total_strategy_runs": len(aggregate_rows),
        "total_evals": int(total_evals),
        "total_time_sec": float(total_time),
        "total_time_min": float(total_time/60),
        "seeds_used": sorted(df_agg["seed"].unique().tolist()),
        "strategies": sorted(df_agg["strategy"].unique().tolist()),
        "datasets": sorted(df_agg["dataset"].unique().tolist()),
    },
    "strategy_comparison": {
        "synthetic_pop20_gen10": synth_comp,
        "fashion_mnist_pop10_gen5": fmnist_comp,
    },
    "sensitivity": {
        "population": {str(k): v for k,v in pop_pivot.to_dict().items()} if not pop_pivot.empty else {},
        "generations": {str(k): v for k,v in gen_pivot.to_dict().items()} if not gen_pivot.empty else {},
        "alpha": {str(k): v for k,v in alpha_pivot.to_dict().items()} if not alpha_pivot.empty else {},
        "k": {str(k): v for k,v in k_pivot.to_dict().items()} if not k_pivot.empty else {},
        "tau": {str(k): v for k,v in tau_pivot.to_dict().items()} if not tau_pivot.empty else {},
    },
    "proxy": {
        "mean_spearman": float(proxy_vals.mean()) if len(proxy_vals) else None,
        "std_spearman": float(proxy_vals.std()) if len(proxy_vals) else None,
        "min_spearman": float(proxy_vals.min()) if len(proxy_vals) else None,
        "max_spearman": float(proxy_vals.max()) if len(proxy_vals) else None,
        "n": int(len(proxy_vals)),
    },
    "multi_seed": {
        "baseline_alpha0_5": ms_baseline_stats,
        "alpha0_05": ms_alpha05_stats,
    },
    "findings": {
        # will be filled manually below after inspection
    },
    "failures": {
        "original_fail_count": 7,
        "fixed_bug_1": "BatchNorm batch_size 1 -> skip batch",
        "fixed_bug_2": "JSON int64 serialization -> default=str + int conversion",
        "final_fail": 0
    }
}

# Findings textual based on data
# Compute dynamic vs random etc across all
# For synthetic baseline pop10 gen5 alpha0.5:
baseline = df_agg[df_agg["experiment"]=="generations/gen_5"]
# dynamic vs random hv
# We'll write hallazgos based on observed numbers, not generic
with open(ROOT / "analysis.json", "w") as f:
    json.dump(analysis, f, indent=2, default=str)
print("Wrote analysis.json")

# Also write a markdown-like findings for dashboard inclusion later
