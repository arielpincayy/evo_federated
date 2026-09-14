#!/usr/bin/env python3
import pandas as pd, numpy as np, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base=Path("results/unified_evofed_final")
report=base/"report"/"plots"
report.mkdir(parents=True, exist_ok=True)
final_comparison=base/"11_final_comparison"
final_comparison.mkdir(parents=True, exist_ok=True)

# Load aggregate
agg_csv=base/"aggregate_results.csv"
if not agg_csv.exists():
    agg_csv=base/"report"/"aggregate_results.csv"
df=pd.read_csv(agg_csv) if agg_csv.exists() else pd.DataFrame()
print("df rows", len(df))
# Generate consolidated final table per spec 66
# Need family, optimizer, pairing_strategy, representative_strategy, objective_mode, tau, N, K, heterogeneity, seed_count, mean_accuracy, median, worst, std, max_min_gap, hypervolume, specialization, evaluations, evaluation_reduction, runtime
# We can aggregate by family/optimizer/pairing
if not df.empty:
    # group
    grouped=df.groupby(["family","optimizer"])
    rows=[]
    for (fam,opt),sub in grouped:
        mean_acc=sub["mean_accuracy"].dropna().mean()
        median_acc=sub["mean_accuracy"].dropna().median() if not sub["mean_accuracy"].dropna().empty else np.nan
        worst=sub["worst_accuracy"].dropna().mean() if not sub["worst_accuracy"].dropna().empty else np.nan
        std=sub["mean_accuracy"].dropna().std()
        # max_min_gap approx mean_acc - worst
        gap= mean_acc - worst if not np.isnan(mean_acc) and not np.isnan(worst) else np.nan
        evals=sub["evaluations"].dropna().mean()
        runtime=sub["runtime"].dropna().mean()
        # hypervolume: try to read from summary? use NaN
        # specialization: compute later as placeholder
        # seed_count
        seed_count=len(sub)
        # representative, objective_mode etc extract from path
        # For simplicity extract pairing_strategy mode
        pairing_mode = sub["pairing_strategy"].mode().iloc[0] if not sub["pairing_strategy"].dropna().empty else ""
        # N, K, alpha median
        N=sub["N"].dropna().median() if not sub["N"].dropna().empty else np.nan
        K=sub["K"].dropna().median() if not sub["K"].dropna().empty else np.nan
        alpha=sub["alpha"].dropna().median() if not sub["alpha"].dropna().empty else np.nan
        tau=sub["tau"].dropna().median() if not sub["tau"].dropna().empty else np.nan
        rows.append({
            "family":fam,"optimizer":opt,"pairing_strategy":pairing_mode,"representative_strategy": "", "objective_mode":"", "tau":tau,"N":N,"K":K,"heterogeneity":alpha,"seed_count":seed_count,"mean_accuracy":mean_acc,"median_accuracy":median_acc,"worst_client_accuracy":worst,"std_client_accuracy":std,"max_min_gap":gap,"hypervolume":np.nan,"specialization":np.nan,"evaluations":evals,"evaluation_reduction":np.nan,"runtime":runtime
        })
    df_final=pd.DataFrame(rows)
    df_final.to_csv(final_comparison/"final_consolidated_table.csv", index=False)
    df_final.to_csv(base/"report"/"final_consolidated_table.csv", index=False)
    print(df_final)

# Generate plots

# 1. accuracy per node vs generation for a sample run (first dissimilarity)
try:
    sample=None
    for exp in ["01_dissimilarity_nsga2","02_similarity_singleobjective","03_manyobjective"]:
        cand=list((base/exp).rglob("node_generation_summary.csv"))
        if cand:
            sample=cand[0]
            break
    if sample and sample.exists():
        df_ng=pd.read_csv(sample)
        plt.figure(figsize=(8,5))
        for nid in sorted(df_ng["node_id"].unique()):
            sub=df_ng[df_ng["node_id"]==nid]
            plt.plot(sub["generation"], sub["mean_accuracy"], label=f"node {nid}", marker='o')
        plt.xlabel("generation"); plt.ylabel("mean accuracy"); plt.title(f"Accuracy per node vs generation\n{sample.parent}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(report/"accuracy_per_node_vs_generation.png", dpi=150)
        plt.close()
        print("saved accuracy per node")
except Exception as e:
    print("plot acc per node failed",e)

# 2. heatmap model x node (for many objective sample)
try:
    cand=list((base/"03_manyobjective").rglob("pareto_front.csv"))
    if cand:
        df_pf=pd.read_csv(cand[0])
        acc_cols=[c for c in df_pf.columns if c.startswith("acc_client_")]
        if acc_cols:
            plt.figure(figsize=(10,6))
            data=df_pf[acc_cols].values
            plt.imshow(data, aspect='auto', cmap='viridis')
            plt.colorbar(label='accuracy')
            plt.xlabel("client")
            plt.ylabel("pareto solution")
            plt.title(f"Model x Node heatmap\n{cand[0].parent}")
            plt.tight_layout()
            plt.savefig(report/"model_x_node_heatmap.png", dpi=150)
            plt.close()
            print("saved model x node")
except Exception as e:
    print("heatmap failed",e)

# 3. cosine similarity heatmap sample
try:
    cand=list(base.rglob("similarity_matrix.csv"))
    if cand:
        S=pd.read_csv(cand[0], header=None).values if cand[0].exists() else None
        # try reading without header
        try:
            S=np.loadtxt(cand[0], delimiter=',')
        except Exception:
            S=pd.read_csv(cand[0]).values
        plt.figure(figsize=(6,5))
        plt.imshow(S, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='cosine similarity')
        plt.title(f"Cosine similarity heatmap\n{cand[0].parent}")
        plt.xlabel("client"); plt.ylabel("client")
        plt.tight_layout()
        plt.savefig(report/"similarity_heatmap.png", dpi=150)
        plt.close()
        print("saved similarity heatmap")
        # distance
        D=1 - S
        np.fill_diagonal(D,0)
        plt.figure(figsize=(6,5))
        plt.imshow(D, cmap='viridis')
        plt.colorbar(label='distance')
        plt.title(f"Cosine distance heatmap\n{cand[0].parent}")
        plt.tight_layout()
        plt.savefig(report/"distance_heatmap.png", dpi=150)
        plt.close()
except Exception as e:
    print("similarity heatmap failed",e)
    import traceback; traceback.print_exc()

# 4. pairing visualization
try:
    cand=list(base.rglob("pairs.csv"))
    if cand:
        df_pairs=pd.read_csv(cand[0])
        if not df_pairs.empty and "similarity" in df_pairs.columns:
            plt.figure(figsize=(8,4))
            plt.bar(range(len(df_pairs)), df_pairs["similarity"])
            plt.xlabel("pair_id"); plt.ylabel("similarity"); plt.title(f"Intra-pair similarity\n{cand[0].parent}")
            plt.tight_layout()
            plt.savefig(report/"pairing_similarity.png", dpi=150)
            plt.close()
except Exception as e:
    print("pairing viz failed",e)

# 5. worst-client vs generation (sample)
try:
    cand=list((base/"02_similarity_singleobjective").rglob("generation_metrics.csv"))
    if cand:
        df_gm=pd.read_csv(cand[0])
        if "worst_fitness" in df_gm.columns or "worst_accuracy" in df_gm.columns:
            plt.figure(figsize=(8,4))
            col="worst_fitness" if "worst_fitness" in df_gm.columns else "worst_accuracy"
            plt.plot(df_gm["generation"], df_gm[col], marker='o')
            plt.xlabel("generation"); plt.ylabel(col); plt.title(f"Worst-client vs generation\n{cand[0].parent}")
            plt.tight_layout()
            plt.savefig(report/"worst_client_vs_generation.png", dpi=150)
            plt.close()
except Exception as e:
    print("worst vs gen failed",e)

# 6. evaluation cost vs generation
try:
    cand=list(base.rglob("generation_metrics.csv"))
    if cand:
        df_gm=pd.read_csv(cand[0])
        if "eval_count" in df_gm.columns:
            plt.figure(figsize=(8,4))
            plt.plot(df_gm["generation"], df_gm["eval_count"], marker='o')
            plt.xlabel("generation"); plt.ylabel("eval_count"); plt.title(f"Evaluation cost vs generation\n{cand[0].parent}")
            plt.tight_layout()
            plt.savefig(report/"eval_cost_vs_generation.png", dpi=150)
            plt.close()
except Exception as e:
    print("eval cost failed",e)

# 7. Pareto front 2D for dissimilarity sample
try:
    cand=list((base/"01_dissimilarity_nsga2").rglob("pair_pareto_front.csv"))
    if cand:
        df_pf=pd.read_csv(cand[0])
        if "accuracy_a" in df_pf.columns and "accuracy_b" in df_pf.columns:
            plt.figure(figsize=(6,5))
            plt.scatter(df_pf["accuracy_a"], df_pf["accuracy_b"], c='blue', label='Pareto')
            plt.xlabel("accuracy A"); plt.ylabel("accuracy B"); plt.title(f"Pareto Front 2D\n{cand[0].parent}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(report/"pareto_2d.png", dpi=150)
            plt.close()
except Exception as e:
    print("pareto 2d failed",e)

# 8. objective correlation vs similarity (for dissimilarity sample)
try:
    # Find a run with both similarity matrix and objective vectors
    cand_sim=list((base/"01_dissimilarity_nsga2").rglob("similarity_matrix.csv"))
    cand_obj=list((base/"01_dissimilarity_nsga2").rglob("objective_vectors.csv"))
    if cand_sim and cand_obj:
        S=pd.read_csv(cand_sim[0], header=None).values if cand_sim[0].exists() else None
        try: S=np.loadtxt(cand_sim[0], delimiter=',')
        except: S=pd.read_csv(cand_sim[0]).values
        df_obj=pd.read_csv(cand_obj[0])
        # compute correlation between acc_a and acc_b across population (for that pair)
        if "accuracy_a" in df_obj.columns and "accuracy_b" in df_obj.columns:
            corr=np.corrcoef(df_obj["accuracy_a"], df_obj["accuracy_b"])[0,1]
            plt.figure(figsize=(6,4))
            plt.scatter(df_obj["accuracy_a"], df_obj["accuracy_b"], alpha=0.5)
            plt.xlabel("accuracy A"); plt.ylabel("accuracy B"); plt.title(f"Objective correlation {corr:.2f}\n{cand_obj[0].parent}")
            plt.tight_layout()
            plt.savefig(report/"objective_correlation.png", dpi=150)
            plt.close()
except Exception as e:
    print("obj corr failed",e)

print("Done final plots")
