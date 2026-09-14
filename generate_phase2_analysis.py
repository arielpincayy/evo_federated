#!/usr/bin/env python3
"""Generate Phase 2 aggregated analysis and plots."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("results/similarity_pairing/phase2_experiments")
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)

def collect_summaries():
    rows = []
    for p in BASE.rglob("summary.json"):
        # exclude phase1 etc.
        if "phase2_experiments" not in str(p):
            continue
        # Determine experiment group
        rel = p.relative_to(BASE)
        parts = rel.parts
        # e.g., A_deterministic_similarity/seed_042/summary.json
        # or C_probabilistic_tau/tau_0_01/seed_042/summary.json
        # or D_node_scaling/n_4/seed_042/summary.json
        exp = parts[0]
        try:
            with open(p) as f:
                s = json.load(f)
        except Exception:
            continue
        # Extract tau/n/alpha etc from path
        tau = None
        n = None
        alpha = None
        k = None
        rep = None
        if "tau" in str(p):
            # find tau_* folder
            for part in parts:
                if part.startswith("tau_"):
                    tau_str = part.replace("tau_","").replace("_",".")
                    try:
                        tau = float(tau_str)
                    except:
                        tau = tau_str
        if "n_" in str(p):
            for part in parts:
                if part.startswith("n_"):
                    try:
                        n = int(part.replace("n_",""))
                    except:
                        pass
        if "alpha" in str(p):
            for part in parts:
                if part.startswith("alpha_"):
                    a_str = part.replace("alpha_","").replace("_",".")
                    try:
                        alpha = float(a_str)
                    except:
                        alpha = a_str
        if "K_" in str(p):
            for part in parts:
                if part.startswith("K_"):
                    try:
                        k = int(part.replace("K_",""))
                    except:
                        pass
        if "rep_" in str(p):
            for part in parts:
                if part.startswith("rep_"):
                    rep = part.replace("rep_","")

        seed = s.get("seed")
        # full flag
        full_mode = s.get("full_mode", False)
        pairing_mode = s.get("pairing_mode")
        rows.append({
            "exp": exp,
            "path": str(p),
            "seed": seed,
            "final_mean_accuracy": s.get("final_mean_accuracy"),
            "final_worst_accuracy": s.get("final_worst_accuracy"),
            "final_std_accuracy": s.get("final_std_accuracy"),
            "best_fitness": s.get("best_fitness_evolution"),
            "n_reps": s.get("n_reps"),
            "total_search_evals": s.get("cost", {}).get("total_search_evaluations"),
            "theoretical_full_evals": s.get("cost", {}).get("theoretical_full_evaluations"),
            "reduction_percent": s.get("cost", {}).get("reduction_percent_vs_full"),
            "total_time": s.get("cost", {}).get("total_time_sec"),
            "evolution_time": s.get("cost", {}).get("evolution_time_sec"),
            "pairing_mode": pairing_mode,
            "tau": tau,
            "n_clients": n if n is not None else s.get("cost", {}).get("n_clients"),
            "alpha": alpha,
            "k_epochs": k if k is not None else s.get("cost", {}).get("characterization_k_epochs"),
            "rep_mode": rep,
            "full_mode": full_mode,
            "prob_entropy": s.get("prob_info", {}).get("entropy_mean") if s.get("prob_info") else None,
        })
    df = pd.DataFrame(rows)
    return df

df = collect_summaries()
print(f"Collected {len(df)} summaries")
print(df["exp"].value_counts())
df.to_csv(OUT / "all_summaries.csv", index=False)

# Helper for stats
def compute_stats(group):
    vals = group["final_mean_accuracy"].dropna()
    worsts = group["final_worst_accuracy"].dropna()
    if len(vals)==0:
        return {}
    mean = float(vals.mean())
    median = float(vals.median())
    std = float(vals.std(ddof=1)) if len(vals)>1 else 0.0
    # 95% CI using normal approx
    ci = 1.96*std/np.sqrt(len(vals)) if len(vals)>1 else 0.0
    return {
        "n": int(len(vals)),
        "mean_accuracy": mean,
        "median_accuracy": median,
        "std_accuracy": std,
        "ci95": float(ci),
        "ci_low": float(mean-ci),
        "ci_high": float(mean+ci),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean_worst": float(worsts.mean()) if len(worsts) else 0,
        "std_worst": float(worsts.std(ddof=1)) if len(worsts)>1 else 0,
        "median_worst": float(worsts.median()) if len(worsts) else 0,
        "mean_reduction": float(group["reduction_percent"].mean()) if "reduction_percent" in group else 0,
        "mean_evals": float(group["total_search_evals"].mean()) if "total_search_evals" in group else 0,
    }

# Experiment A vs B vs H comparison (deterministic vs random vs full)
def stats_for_exp(exp_name):
    sub = df[df["exp"]==exp_name]
    if sub.empty:
        return None
    return compute_stats(sub)

# A deterministic
stats_A = stats_for_exp("A_deterministic_similarity")
stats_B = stats_for_exp("B_random_pairing")
stats_H = stats_for_exp("H_full_baseline")
print("\nA deterministic", stats_A)
print("B random", stats_B)
print("H full", stats_H)

# Save table
ablation_rows = []
for exp in ["A_deterministic_similarity","B_random_pairing","H_full_baseline"]:
    st = stats_for_exp(exp)
    if st:
        ablation_rows.append({"exp":exp, **st})
pd.DataFrame(ablation_rows).to_csv(OUT / "tables" / "ablation_deterministic_random_full.csv", index=False)

# Statistical comparison: effect size and simple t-test if enough seeds
try:
    from scipy import stats as scipy_stats
    # A vs B
    a_vals = df[df["exp"]=="A_deterministic_similarity"]["final_mean_accuracy"].dropna().values
    b_vals = df[df["exp"]=="B_random_pairing"]["final_mean_accuracy"].dropna().values
    h_vals = df[df["exp"]=="H_full_baseline"]["final_mean_accuracy"].dropna().values
    if len(a_vals) and len(b_vals):
        # paired? seeds matched? Use same seeds set, but we have 10 each same seeds list, so paired per seed
        # Merge on seed
        merged = pd.merge(df[df["exp"]=="A_deterministic_similarity"][["seed","final_mean_accuracy"]].rename(columns={"final_mean_accuracy":"a"}),
                          df[df["exp"]=="B_random_pairing"][["seed","final_mean_accuracy"]].rename(columns={"final_mean_accuracy":"b"}),
                          on="seed")
        if not merged.empty:
            from scipy.stats import wilcoxon, ttest_rel
            diff = merged["a"] - merged["b"]
            try:
                w_stat, w_p = wilcoxon(merged["a"], merged["b"])
            except Exception:
                w_stat, w_p = None, None
            try:
                t_stat, t_p = ttest_rel(merged["a"], merged["b"])
            except Exception:
                t_stat, t_p = None, None
            cohen_d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1)!=0 else 0
            print(f"\nA vs B paired diff mean {diff.mean():.4f} std {diff.std():.4f} d {cohen_d:.2f} t_p {t_p} w_p {w_p}")
            with open(OUT / "statistics_A_vs_B.json","w") as f:
                json.dump({"mean_diff": float(diff.mean()), "std_diff": float(diff.std()), "cohen_d": float(cohen_d), "t_p": float(t_p) if t_p is not None else None, "w_p": float(w_p) if w_p is not None else None, "n": len(diff)}, f, indent=2)
    # A vs H similarly
    merged_h = pd.merge(df[df["exp"]=="A_deterministic_similarity"][["seed","final_mean_accuracy"]].rename(columns={"final_mean_accuracy":"a"}),
                        df[df["exp"]=="H_full_baseline"][["seed","final_mean_accuracy"]].rename(columns={"final_mean_accuracy":"h"}),
                        on="seed")
    if not merged_h.empty:
        diff_h = merged_h["a"] - merged_h["h"]
        try:
            w_stat, w_p = scipy_stats.wilcoxon(merged_h["a"], merged_h["h"])
        except:
            w_stat, w_p = None, None
        try:
            t_stat, t_p = scipy_stats.ttest_rel(merged_h["a"], merged_h["h"])
        except:
            t_stat, t_p = None, None
        cohen_d = float(diff_h.mean() / diff_h.std(ddof=1)) if diff_h.std(ddof=1)!=0 else 0
        print(f"A vs H diff mean {diff_h.mean():.4f} d {cohen_d:.2f} t_p {t_p} w_p {w_p}")
        with open(OUT / "statistics_A_vs_H.json","w") as f:
            json.dump({"mean_diff": float(diff_h.mean()), "std_diff": float(diff_h.std()), "cohen_d": float(cohen_d), "t_p": float(t_p) if t_p is not None else None, "w_p": float(w_p) if w_p is not None else None, "n": len(diff_h)}, f, indent=2)
except Exception as e:
    print(f"stats failed {e}")
    import traceback
    traceback.print_exc()

# C tau sweep
tau_df = df[df["exp"]=="C_probabilistic_tau"]
if not tau_df.empty:
    tau_stats = []
    for tau, sub in tau_df.groupby("tau"):
        st = compute_stats(sub)
        tau_stats.append({"tau": tau, **st})
    tau_stats = sorted(tau_stats, key=lambda x: x["tau"])
    pd.DataFrame(tau_stats).to_csv(OUT / "tables" / "tau_sweep.csv", index=False)
    print("\nTau sweep", tau_stats[:3])
    # plot
    plt.figure(figsize=(8,5))
    taus = [x["tau"] for x in tau_stats]
    means = [x["mean_accuracy"] for x in tau_stats]
    stds = [x["std_accuracy"] for x in tau_stats]
    plt.errorbar(taus, means, yerr=stds, marker="o", capsize=4)
    plt.xscale("log")
    plt.xlabel("Tau")
    plt.ylabel("Mean final accuracy")
    plt.title("Probabilistic pairing: effect of tau")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "tau_vs_accuracy.png", dpi=150)
    plt.close()
    # also worst
    plt.figure(figsize=(8,5))
    worsts = [x["mean_worst"] for x in tau_stats]
    plt.errorbar(taus, worsts, yerr=[x["std_worst"] for x in tau_stats], marker="o", capsize=4, color="red")
    plt.xscale("log")
    plt.xlabel("Tau")
    plt.ylabel("Worst accuracy")
    plt.title("Tau vs worst-client accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "tau_vs_worst.png", dpi=150)
    plt.close()
    # entropy vs tau
    # need to collect entropy from probabilistic runs: already in prob_entropy column but per seed average
    # group mean entropy
    ent = tau_df.groupby("tau")["prob_entropy"].mean().reset_index()
    if not ent.empty:
        plt.figure(figsize=(8,4))
        plt.plot(ent["tau"], ent["prob_entropy"], marker="o")
        plt.xscale("log")
        plt.xlabel("Tau")
        plt.ylabel("Mean entropy")
        plt.title("Entropy vs tau (probabilistic pairing)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "plots" / "tau_vs_entropy.png", dpi=150)
        plt.close()

# D node scaling
node_df = df[df["exp"]=="D_node_scaling"]
if not node_df.empty:
    node_stats = []
    for n, sub in node_df.groupby("n_clients"):
        st = compute_stats(sub)
        # add cost
        mean_time = float(sub["total_time"].mean())
        mean_evals = float(sub["total_search_evals"].mean())
        node_stats.append({"n_clients": int(n), **st, "mean_time": mean_time, "mean_evals": mean_evals})
    node_stats = sorted(node_stats, key=lambda x: x["n_clients"])
    pd.DataFrame(node_stats).to_csv(OUT / "tables" / "node_scaling.csv", index=False)
    print("\nNode scaling", node_stats)
    plt.figure(figsize=(8,5))
    ns = [x["n_clients"] for x in node_stats]
    means = [x["mean_accuracy"] for x in node_stats]
    stds = [x["std_accuracy"] for x in node_stats]
    plt.errorbar(ns, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("N clients")
    plt.ylabel("Mean accuracy")
    plt.title("Scaling with number of nodes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "n_scaling_accuracy.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8,5))
    worsts = [x["mean_worst"] for x in node_stats]
    plt.errorbar(ns, worsts, yerr=[x["std_worst"] for x in node_stats], marker="o", capsize=4, color="red")
    plt.xlabel("N")
    plt.ylabel("Worst accuracy")
    plt.title("Worst vs N")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "n_scaling_worst.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8,5))
    evals = [x["mean_evals"] for x in node_stats]
    plt.plot(ns, evals, marker="o", label="similarity")
    # theoretical full
    full_evals = [x["n_clients"]*10*8 for x in node_stats]  # pop10 gen8
    plt.plot(ns, full_evals, marker="s", label="full")
    plt.xlabel("N")
    plt.ylabel("Evals")
    plt.title("Evals vs N")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "n_scaling_evals.png", dpi=150)
    plt.close()

# E heterogeneity
het_df = df[df["exp"]=="E_heterogeneity"]
if not het_df.empty:
    het_stats = []
    for a, sub in het_df.groupby("alpha"):
        st = compute_stats(sub)
        het_stats.append({"alpha": float(a), **st})
    het_stats = sorted(het_stats, key=lambda x: x["alpha"])
    pd.DataFrame(het_stats).to_csv(OUT / "tables" / "heterogeneity.csv", index=False)
    print("\nHeterogeneity", het_stats)
    plt.figure(figsize=(8,5))
    alphas = [x["alpha"] for x in het_stats]
    means = [x["mean_accuracy"] for x in het_stats]
    stds = [x["std_accuracy"] for x in het_stats]
    plt.errorbar(alphas, means, yerr=stds, marker="o", capsize=4)
    plt.xscale("log")
    plt.xlabel("Alpha (Dirichlet)")
    plt.ylabel("Mean accuracy")
    plt.title("Heterogeneity effect")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "alpha_vs_accuracy.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8,5))
    worsts = [x["mean_worst"] for x in het_stats]
    plt.errorbar(alphas, worsts, yerr=[x["std_worst"] for x in het_stats], marker="o", capsize=4, color="red")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Worst accuracy")
    plt.title("Worst vs heterogeneity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "alpha_vs_worst.png", dpi=150)
    plt.close()

# F K
k_df = df[df["exp"]=="F_exploratory_K"]
if not k_df.empty:
    k_stats = []
    for k, sub in k_df.groupby("k_epochs"):
        st = compute_stats(sub)
        k_stats.append({"k": int(k), **st})
    k_stats = sorted(k_stats, key=lambda x: x["k"])
    pd.DataFrame(k_stats).to_csv(OUT / "tables" / "k_sweep.csv", index=False)
    print("\nK sweep", k_stats)
    plt.figure(figsize=(8,5))
    ks = [x["k"] for x in k_stats]
    means = [x["mean_accuracy"] for x in k_stats]
    stds = [x["std_accuracy"] for x in k_stats]
    plt.errorbar(ks, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("K exploratory epochs")
    plt.ylabel("Mean accuracy")
    plt.title("Effect of K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "K_vs_accuracy.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8,5))
    worsts = [x["mean_worst"] for x in k_stats]
    plt.errorbar(ks, worsts, yerr=[x["std_worst"] for x in k_stats], marker="o", capsize=4, color="red")
    plt.xlabel("K")
    plt.ylabel("Worst accuracy")
    plt.title("Worst vs K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "K_vs_worst.png", dpi=150)
    plt.close()
    # also cost for K (characterization time already low)

# G rep strategy
rep_df = df[df["exp"]=="G_rep_strategy"]
if not rep_df.empty:
    rep_stats = []
    for rep, sub in rep_df.groupby("rep_mode"):
        st = compute_stats(sub)
        rep_stats.append({"rep": rep, **st})
    pd.DataFrame(rep_stats).to_csv(OUT / "tables" / "rep_strategy.csv", index=False)
    print("\nRep strategy", rep_stats)
    plt.figure(figsize=(6,4))
    labels = [x["rep"] for x in rep_stats]
    means = [x["mean_accuracy"] for x in rep_stats]
    stds = [x["std_accuracy"] for x in rep_stats]
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Mean accuracy")
    plt.title("Representative strategy")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "rep_strategy.png", dpi=150)
    plt.close()

# Overall comparison: A vs B vs H boxplot
plt.figure(figsize=(8,5))
data = []
labels = []
for exp, vals in [("Deterministic", df[df["exp"]=="A_deterministic_similarity"]["final_mean_accuracy"].dropna().values),
                  ("Random", df[df["exp"]=="B_random_pairing"]["final_mean_accuracy"].dropna().values),
                  ("Full", df[df["exp"]=="H_full_baseline"]["final_mean_accuracy"].dropna().values)]:
    if len(vals):
        data.append(vals)
        labels.append(exp)
if data:
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Final mean accuracy")
    plt.title("A vs B vs H (10 seeds each)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "compare_A_B_H_boxplot.png", dpi=150)
    plt.close()

# Also previous baseline I: try to load summary.csv for I
import pathlib as pl
i_rows = []
for p in (BASE / "I_baseline_previous").rglob("summary.csv"):
    try:
        d = pd.read_csv(p)
        # summary.csv has hv etc., not directly comparable. We'll try to parse
        # But extract mean accuracy? The old run's summary has mean accuracy? Let's check columns
        # For old runner, summary.csv has hv etc., but not final accuracy? Instead check exhaustive
        # Let's look for exhaustive_validation.csv
        pass
    except Exception:
        pass
# Alternative: look for final evaluation in I?
for p in (BASE / "I_baseline_previous").rglob("exhaustive_validation.csv"):
    try:
        d =pd.read_csv(p)
        # each row is genome, take best mean_f1? This is not directly accuracy but we can approximate
        pass
    except Exception:
        pass

# Cost comparison overall
plt.figure(figsize=(8,5))
# mean evals vs mean accuracy scatter for all exps
plt.scatter(df["total_search_evals"], df["final_mean_accuracy"], alpha=0.4, s=20)
plt.xlabel("Search evals")
plt.ylabel("Mean accuracy")
plt.title("Cost vs quality (all runs)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "plots" / "cost_vs_quality.png", dpi=150)
plt.close()

# Similarity analysis: correlation S vs TV across runs
# Collect proxy_validation.json per run
proxy_rows = []
for p in BASE.rglob("proxy_validation.json"):
    try:
        with open(p) as f:
            j=json.load(f)
        # find alpha from path
        # Use pearson_distance_vs_TV
        proxy_rows.append({
            "pearson_D_TV": j.get("pearson_distance_vs_TV"),
            "pearson_S_invTV": j.get("pearson_similarity_vs_invTV"),
            "path": str(p),
        })
    except Exception:
        pass
if proxy_rows:
    pdf = pd.DataFrame(proxy_rows)
    pdf.to_csv(OUT / "proxy_correlation.csv", index=False)
    print(f"\nProxy rows {len(proxy_rows)} mean Pearson D-TV {pdf['pearson_D_TV'].mean():.3f}")

print("\nAnalysis completed. Output at", OUT)
