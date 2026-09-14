#!/usr/bin/env python3
"""Análisis agregado Fase2 many-objective."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "src")
plt.style.use("seaborn-v0_8")

ROOT = Path("results/many_objective_pca/phase2_experiments")
SUMMARY = ROOT / "phase2_summary.csv"
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SUMMARY)
print(f"Loaded {len(df)} runs")
# Add helper: pca simplified
df["pca_simple"] = df["pca_mode"].apply(lambda x: "pca" if x != "none" else "none")
# Also need to parse tau, alpha etc already in df

def stats(group):
    hv = group["hv"]
    mean_best = group["mean_best"]
    worst_best = group["worst_best"]
    return pd.Series({
        "n": len(group),
        "mean_hv": hv.mean(),
        "median_hv": hv.median(),
        "std_hv": hv.std(ddof=1) if len(hv)>1 else 0.0,
        "ci95": 1.96*hv.std(ddof=1)/np.sqrt(len(hv)) if len(hv)>1 else 0.0,
        "mean_acc": mean_best.mean(),
        "median_acc": mean_best.median(),
        "worst_mean": worst_best.mean(),
        "mean_pearson": group["pearson"].mean() if "pearson" in group else 0,
    })

# 1. Baseline factorial optimizer x PCA (A)
baseline = df[df["exp_id"]=="A_baseline"]
if not baseline.empty:
    tbl = baseline.groupby(["optimizer","pca_mode"]).apply(stats).reset_index()
    # also grouped simpler
    tbl2 = baseline.groupby(["optimizer","pca_simple"]).apply(stats).reset_index()
    tbl.to_csv(OUT/"tables/baseline_optimizer_pca.csv", index=False)
    print("Baseline")
    print(tbl.to_string(index=False))
    # plot
    plt.figure(figsize=(8,5))
    labels=[]
    data=[]
    for (opt,pca), grp in baseline.groupby(["optimizer","pca_mode"]):
        labels.append(f"{opt}\n{pca}")
        data.append(grp["hv"].values)
    plt.boxplot(data, labels=labels)
    plt.ylabel("HV pareto")
    plt.title("Baseline HV: MOEAD vs NSGA3 × PCA (10 seeds)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT/"plots/baseline_hv_boxplot.png", dpi=150)
    plt.close()
    # worst accuracy box
    plt.figure(figsize=(8,5))
    data=[]
    for (opt,pca), grp in baseline.groupby(["optimizer","pca_mode"]):
        data.append(grp["worst_best"].values)
    plt.boxplot(data, labels=labels)
    plt.ylabel("Worst-client accuracy (best in pareto)")
    plt.title("Baseline worst accuracy")
    plt.tight_layout()
    plt.savefig(OUT/"plots/baseline_worst_boxplot.png", dpi=150)
    plt.close()
    # cost vs hv scatter
    # need time from runtime? df doesn't have time, use mean hv vs D_red? Skip

# 2. Tau sweep B
tau_df = df[df["exp_id"]=="B_tau"]
if not tau_df.empty:
    # tau column is string maybe, convert
    tau_df["tau"] = pd.to_numeric(tau_df["tau"], errors="coerce")
    tbl = tau_df.groupby(["optimizer","pca_mode","tau"]).apply(stats).reset_index()
    tbl.to_csv(OUT/"tables/tau_sweep.csv", index=False)
    print("\nTau sweep (first 10)")
    print(tbl.head(10).to_string(index=False))
    # For each optimizer x pca, plot hv vs tau (should be flat since independent)
    for (opt,pca), grp in tau_df.groupby(["optimizer","pca_mode"]):
        agg = grp.groupby("tau")["hv"].mean()
        plt.figure(figsize=(6,4))
        plt.plot(agg.index, agg.values, marker="o")
        plt.xscale("log")
        plt.xlabel("tau")
        plt.ylabel("HV mean")
        plt.title(f"HV vs tau ({opt} {pca})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"plots/tau_vs_hv_{opt}_{pca}.png", dpi=150)
        plt.close()
        # also pearson preservation vs tau (should not vary, since tau doesn't affect similarity)
        # but preservation is per run same, so flat as well
        # instead we show entropy vs tau from probabilistic pairing info? Need to read probabilistic_pairing.json per run
    # Overall tau preservation vs hv (scatter)
    # Try to collect entropy per tau: read per run file
    # For brevity, compute average preservation per tau
    pres_by_tau = tau_df.groupby("tau")["pearson"].mean()
    plt.figure(figsize=(6,4))
    plt.plot(pres_by_tau.index, pres_by_tau.values, marker="o", color="green")
    plt.xscale("log")
    plt.xlabel("tau")
    plt.ylabel("Pearson preservation (avg)")
    plt.title("Similarity preservation vs tau")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"plots/tau_vs_preservation.png", dpi=150)
    plt.close()

# 3. Node scaling C
node_df = df[df["exp_id"].str.startswith("C_n")]
if not node_df.empty:
    # n_clients already in df
    tbl = node_df.groupby(["n_clients","optimizer","pca_mode"]).apply(stats).reset_index()
    tbl.to_csv(OUT/"tables/node_scaling.csv", index=False)
    print("\nNode scaling")
    print(tbl.head(20).to_string(index=False))
    for (opt,pca), grp in node_df.groupby(["optimizer","pca_mode"]):
        agg = grp.groupby("n_clients")["hv"].mean()
        plt.figure(figsize=(6,4))
        plt.plot(agg.index, agg.values, marker="o")
        plt.xlabel("N clients (= objectives)")
        plt.ylabel("HV mean")
        plt.title(f"HV vs N ({opt} {pca})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"plots/node_scaling_hv_{opt}_{pca}.png", dpi=150)
        plt.close()
        # worst accuracy vs N
        agg2 = grp.groupby("n_clients")["worst_best"].mean()
        plt.figure(figsize=(6,4))
        plt.plot(agg2.index, agg2.values, marker="s", color="orange")
        plt.xlabel("N")
        plt.ylabel("Worst accuracy")
        plt.title(f"Worst vs N ({opt} {pca})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"plots/node_scaling_worst_{opt}_{pca}.png", dpi=150)
        plt.close()
    # also time vs N: need to read runtime.json per run for time, but df time missing (we stored hv time? Actually df has no time col, we have D_red etc, need to load runtime)
    # Collect time per N
    rows_time=[]
    for _,row in node_df.iterrows():
        out = Path(row["out"])
        try:
            rt=json.load(open(out/"runtime.json"))
            rows_time.append({"n": row["n_clients"], "opt": row["optimizer"], "pca": row["pca_mode"], "time": rt["total_time"]})
        except: pass
    if rows_time:
        df_time=pd.DataFrame(rows_time)
        agg_t=df_time.groupby(["n","opt","pca"])["time"].mean().reset_index()
        for (opt,pca), grp in df_time.groupby(["opt","pca"]):
            sub=grp.groupby("n")["time"].mean()
            plt.figure(figsize=(6,4))
            plt.plot(sub.index, sub.values, marker="o", color="red")
            plt.xlabel("N")
            plt.ylabel("Runtime (s)")
            plt.title(f"Runtime vs N ({opt} {pca})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT/f"plots/node_scaling_time_{opt}_{pca}.png", dpi=150)
            plt.close()

# 4. Heterogeneity D
het_df = df[df["exp_id"].str.startswith("D_alpha")]
if not het_df.empty:
    tbl = het_df.groupby(["alpha","optimizer","pca_mode"]).apply(stats).reset_index()
    tbl.to_csv(OUT/"tables/heterogeneity.csv", index=False)
    print("\nHeterogeneity")
    print(tbl.head(20).to_string(index=False))
    for (opt,pca), grp in het_df.groupby(["optimizer","pca_mode"]):
        agg = grp.groupby("alpha")["hv"].mean()
        plt.figure(figsize=(6,4))
        plt.plot(agg.index, agg.values, marker="o")
        plt.xscale("log")
        plt.xlabel("alpha (heterogeneity, low=high)")
        plt.ylabel("HV mean")
        plt.title(f"HV vs alpha ({opt} {pca})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"plots/hetero_hv_{opt}_{pca}.png", dpi=150)
        plt.close()
        # worst vs alpha
        agg2 = grp.groupby("alpha")["worst_best"].mean()
        plt.figure(figsize=(6,4))
        plt.plot(agg2.index, agg2.values, marker="s", color="orange")
        plt.xscale("log")
        plt.xlabel("alpha")
        plt.ylabel("Worst accuracy")
        plt.title(f"Worst vs alpha ({opt} {pca})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"plots/hetero_worst_{opt}_{pca}.png", dpi=150)
        plt.close()

# 5. K epochs E
k_df = df[df["exp_id"].str.startswith("E_K")]
if not k_df.empty:
    tbl = k_df.groupby(["k_epochs","optimizer","pca_mode"]).apply(stats).reset_index()
    tbl.to_csv(OUT/"tables/local_epochs.csv", index=False)
    print("\nK epochs")
    print(tbl.to_string(index=False))
    for (opt,pca), grp in k_df.groupby(["optimizer","pca_mode"]):
        agg = grp.groupby("k_epochs")["hv"].mean()
        plt.figure(figsize=(6,4))
        plt.plot(agg.index, agg.values, marker="o")
        plt.xlabel("K local epochs (Δ generation)")
        plt.ylabel("HV mean")
        plt.title(f"HV vs K ({opt} {pca})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"plots/K_hv_{opt}_{pca}.png", dpi=150)
        plt.close()
        # preservation vs K
        aggP = grp.groupby("k_epochs")["pearson"].mean()
        plt.figure(figsize=(6,4))
        plt.plot(aggP.index, aggP.values, marker="s", color="green")
        plt.xlabel("K")
        plt.ylabel("Pearson preservation")
        plt.title(f"Preservation vs K ({opt} {pca})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"plots/K_preservation_{opt}_{pca}.png", dpi=150)
        plt.close()

# 6. PCA components F
pca_df = df[df["exp_id"].str.startswith("F_pca")]
if not pca_df.empty:
    # Need to distinguish variance vs fixed: exp_id contains var or fixed
    # extract variance value or fixed value from exp_id
    def pca_key(row):
        if "var" in row["exp_id"]:
            return row["exp_id"]  # F_pca_var0.9 etc
        else:
            return row["exp_id"]
    tbl = pca_df.groupby(["exp_id","optimizer"]).apply(stats).reset_index()
    tbl.to_csv(OUT/"tables/pca_components.csv", index=False)
    print("\nPCA components")
    print(tbl.to_string(index=False))
    # For each optimizer, plot HV vs components
    for opt, grp in pca_df.groupby("optimizer"):
        # variance group
        var_grp = grp[grp["exp_id"].str.contains("var")]
        if not var_grp.empty:
            # map var value: extract numeric
            var_grp["var_val"] = var_grp["exp_id"].str.extract(r"var([0-9_.]+)").replace("_",".", regex=True).astype(float)
            agg = var_grp.groupby("var_val")["hv"].mean()
            plt.figure(figsize=(6,4))
            plt.plot(agg.index, agg.values, marker="o")
            plt.xlabel("PCA variance target")
            plt.ylabel("HV mean")
            plt.title(f"HV vs PCA variance ({opt})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT/f"plots/pca_variance_hv_{opt}.png", dpi=150)
            plt.close()
        fixed_grp = grp[grp["exp_id"].str.contains("fixed")]
        if not fixed_grp.empty:
            fixed_grp["comp"] = fixed_grp["exp_id"].str.extract(r"fixed(\d+)").astype(int)
            agg = fixed_grp.groupby("comp")["hv"].mean()
            plt.figure(figsize=(6,4))
            plt.plot(agg.index, agg.values, marker="s")
            plt.xlabel("PCA fixed components")
            plt.ylabel("HV mean")
            plt.title(f"HV vs fixed components ({opt})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT/f"plots/pca_fixed_hv_{opt}.png", dpi=150)
            plt.close()
            # preservation vs fixed
            aggP = fixed_grp.groupby("comp")["pearson"].mean()
            plt.figure(figsize=(6,4))
            plt.plot(aggP.index, aggP.values, marker="o", color="green")
            plt.xlabel("components")
            plt.ylabel("Pearson preservation")
            plt.title(f"Preservation vs fixed ({opt})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT/f"plots/pca_fixed_preservation_{opt}.png", dpi=150)
            plt.close()

# 7. Statistical comparisons factorial: optimizer x PCA overall (baseline)
# Compute paired differences per seed for baseline
baseline = df[df["exp_id"]=="A_baseline"]
if not baseline.empty:
    # Pivot to compare moead vs nsga3 for same seed+pca, and pca vs none for same seed+optimizer
    # MOEAD vs NSGA3
    piv = baseline.pivot_table(index=["seed","pca_mode"], columns="optimizer", values="hv").reset_index()
    if "moead" in piv.columns and "nsga3" in piv.columns:
        diff = piv["nsga3"] - piv["moead"]
        from scipy.stats import wilcoxon, ttest_rel
        try:
            w_p = wilcoxon(diff).pvalue if len(diff)>5 else float("nan")
            t_p = ttest_rel(piv["nsga3"], piv["moead"]).pvalue
            cohen = diff.mean() / (diff.std(ddof=1) + 1e-9)
        except: w_p = t_p = cohen = float("nan")
        stats_opt = {
            "mean_moead": float(piv["moead"].mean()),
            "mean_nsga3": float(piv["nsga3"].mean()),
            "mean_diff_nsga3_minus_moead": float(diff.mean()),
            "std_diff": float(diff.std()),
            "cohen_d": float(cohen),
            "wilcoxon_p": float(w_p),
            "t_p": float(t_p),
            "n": int(len(diff))
        }
        with open(OUT/"tables/stats_moead_vs_nsga3.json","w") as f: json.dump(stats_opt,f,indent=2)
        print("\nMOEAD vs NSGA3")
        print(stats_opt)
        # PCA vs none
        piv2 = baseline.pivot_table(index=["seed","optimizer"], columns="pca_mode", values="hv").reset_index()
        # columns are 'none' and 'variance', need to handle naming
        cols = piv2.columns
        if "none" in piv2.columns and "variance" in piv2.columns:
            diff2 = piv2["variance"] - piv2["none"]
            try:
                w2 = wilcoxon(diff2).pvalue if len(diff2)>5 else float("nan")
                t2 = ttest_rel(piv2["variance"], piv2["none"]).pvalue
                co2 = diff2.mean()/(diff2.std(ddof=1)+1e-9)
            except: w2=t2=co2=float("nan")
            stats_pca = {
                "mean_none": float(piv2["none"].mean()),
                "mean_pca": float(piv2["variance"].mean()),
                "mean_diff_pca_minus_none": float(diff2.mean()),
                "std_diff": float(diff2.std()),
                "cohen_d": float(co2),
                "wilcoxon_p": float(w2),
                "t_p": float(t2),
                "n": int(len(diff2))
            }
            with open(OUT/"tables/stats_pca_vs_none.json","w") as f: json.dump(stats_pca,f,indent=2)
            print("\nPCA vs none")
            print(stats_pca)

# 8. Overall summary csv for all experiments average per exp_id optimizer pca
overall = df.groupby(["exp_id","optimizer","pca_mode"]).apply(stats).reset_index()
overall.to_csv(OUT/"tables/overall_summary.csv", index=False)
print("\nOverall summary rows", len(overall))

# 9. Preservation analysis G: use preservation column already, but also compute more detailed per run from similarity matrices?
# For quick, use df pearson already
pres_df = df[df["pca_mode"]!="none"]
if not pres_df.empty:
    print("\nPreservation overall mean pearson", pres_df["pearson"].mean(), "median", pres_df["pearson"].median())
    plt.figure(figsize=(6,4))
    plt.hist(pres_df["pearson"], bins=20, alpha=0.7)
    plt.xlabel("Pearson preservation")
    plt.ylabel("Count")
    plt.title("Histogram preservation (all PCA runs)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"plots/preservation_hist.png", dpi=150)
    plt.close()

print("Analysis done, tables in", OUT/"tables", "plots in", OUT/"plots")
