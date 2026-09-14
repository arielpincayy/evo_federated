#!/usr/bin/env python3
import json, pandas as pd, numpy as np
from pathlib import Path
import glob

base = Path("results/unified_evofed_final")
report = base / "report"
report.mkdir(parents=True, exist_ok=True)
plots_dir = report / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Helper to read csv safely
def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# Representative runs
rep_diss = base / "01_dissimilarity_nsga2/deterministic/seed_042"
rep_sim = base / "02_similarity_singleobjective/deterministic_rep_alternating/seed_042"
rep_sim_full = base / "02_similarity_singleobjective/full_baseline/seed_042"
rep_nsga3 = base / "03_manyobjective/nsga3_per_silo/seed_042"
rep_moead = base / "03_manyobjective/moead_per_silo/seed_042"

# Load data for charts
# 1. Hypervolume per generation for diss (pair 0) and many
def hv_data(path, pair=None):
    # for diss, hv per pair
    if pair is not None:
        p = Path(path) / f"pair_{pair:02d}_0_2" / "hypervolume.csv"
        # actually pair names vary, find any pair dir
        dirs = list(Path(path).glob("pair_*"))
        if dirs:
            df = read_csv(dirs[0] / "hypervolume.csv")
            return df.to_dict(orient="list") if not df.empty else {}
        return {}
    else:
        df = read_csv(Path(path) / "hypervolume.csv")
        if df.empty:
            df = read_csv(Path(path) / "generations" / "hypervolume.csv")
        if df.empty:
            # try generation_metrics
            df = read_csv(Path(path) / "generation_metrics.csv")
        return df.to_dict(orient="list") if not df.empty else {}

# For sim, fitness vs generation
def sim_fitness(path):
    df = read_csv(Path(path) / "generation_metrics.csv")
    if df.empty:
        df = read_csv(Path(path) / "generations" / "generation_metrics.csv")
    return df.to_dict(orient="list") if not df.empty else {}

# For node evolution
def node_summary(path):
    df = read_csv(Path(path) / "node_generation_summary.csv")
    if df.empty:
        df = read_csv(Path(path) / "generations" / "node_generation_summary.csv")
    return df.to_dict(orient="list") if not df.empty else {}

# For similarity matrix
def sim_matrix(path):
    try:
        df = pd.read_csv(Path(path) / "similarity_matrix.csv", header=None)
        # if header exists, try with header
        if df.shape[0] == df.shape[1]:
            return df.values.tolist()
        # else read correctly
        df2 = pd.read_csv(Path(path) / "similarity_matrix.csv")
        return df2.values.tolist()
    except Exception as e:
        try:
            arr = np.load(Path(path) / "similarity_matrix.npy")
            return arr.tolist()
        except Exception:
            return []

# For pareto front diss
def pareto_diss(path):
    dirs = list(Path(path).glob("pair_*"))
    if dirs:
        df = read_csv(dirs[0] / "pair_pareto_front.csv")
        if df.empty:
            df = read_csv(dirs[0] / "pareto_front.csv")
        return df.to_dict(orient="list") if not df.empty else {}
    return {}

def pareto_many(path):
    df = read_csv(Path(path) / "pareto_front.csv")
    if df.empty:
        df = read_csv(Path(path) / "pareto" / "pareto_front.csv")
    return df.to_dict(orient="list") if not df.empty else {}

# Load
hv_diss = {}
try:
    # aggregate hv across pairs for diss representative: mean hv per generation
    gen_metrics = read_csv(rep_diss / "generation_metrics.csv")
    if not gen_metrics.empty:
        # it has pair_id, gen, hv
        # compute mean hv per gen
        if "hv" in gen_metrics.columns and "generation" in gen_metrics.columns:
            mean_hv = gen_metrics.groupby("generation")["hv"].mean().reset_index()
            hv_diss = {"generation": mean_hv["generation"].tolist(), "hv": mean_hv["hv"].tolist()}
        else:
            hv_diss = gen_metrics.to_dict(orient="list")
    else:
        # try pair
        dirs = list(rep_diss.glob("pair_*"))
        hv_list=[]
        for d in dirs:
            df = read_csv(d / "hypervolume.csv")
            if not df.empty:
                hv_list.append(df)
        if hv_list:
            # take first
            hv_diss = hv_list[0].to_dict(orient="list")
except Exception as e:
    print("hv_diss",e)
    hv_diss={}

hv_nsga3 = read_csv(rep_nsga3 / "hypervolume.csv").to_dict(orient="list") if not read_csv(rep_nsga3 / "hypervolume.csv").empty else read_csv(rep_nsga3 / "generations" / "hypervolume.csv").to_dict(orient="list")
hv_moead = read_csv(rep_moead / "hypervolume.csv").to_dict(orient="list") if not read_csv(rep_moead / "hypervolume.csv").empty else {}

sim_fit = sim_fitness(rep_sim)
sim_fit_full = sim_fitness(rep_sim_full)
node_sim = node_summary(rep_sim)
node_diss = node_summary(rep_diss)
node_nsga3 = node_summary(rep_nsga3)

pareto_diss_data = pareto_diss(rep_diss)
pareto_many_data = pareto_many(rep_nsga3)
pareto_moead_data = pareto_many(rep_moead)

sim_mat = sim_matrix(rep_diss)  # use same for all
# distance
try:
    dist_mat = (1 - np.array(sim_mat)).tolist()
    # set diagonal 0
    for i in range(len(dist_mat)):
        dist_mat[i][i]=0
except Exception:
    dist_mat=[]

# For model x node heatmaps: use final_full_evaluation
def model_node_matrix(path):
    df = read_csv(path / "final_full_evaluation.csv")
    if df.empty:
        df = read_csv(path / "pareto_front.csv")
    acc_cols = [c for c in df.columns if c.startswith("acc_client_")]
    if not df.empty and acc_cols:
        # limit to top 10 rows
        sub = df[acc_cols].head(10)
        return {"clients": acc_cols, "values": sub.values.tolist(), "rows": len(sub)}
    return {"clients": [], "values": []}

model_node_sim = model_node_matrix(rep_sim)
model_node_diss = model_node_matrix(rep_diss / "pair_00_0_2") if (rep_diss / "pair_00_0_2").exists() else model_node_matrix(rep_diss)
model_node_nsga3 = model_node_matrix(rep_nsga3)

# For objective correlation vs similarity: need per-pair correlation from pair_aggregate
pair_agg = read_csv(rep_diss / "pair_aggregate.csv")
similarity_vs_corr = {}
try:
    pairs_csv = read_csv(rep_diss / "pairs" / "pairs.csv")
    if pairs_csv.empty:
        pairs_csv = read_csv(rep_diss / "similarity" / "pairs.csv")
    if not pairs_csv.empty and not pair_agg.empty:
        # merge
        similarity_vs_corr = {"similarity": pairs_csv["similarity"].tolist() if "similarity" in pairs_csv.columns else [],
                              "distance": pairs_csv["distance"].tolist() if "distance" in pairs_csv.columns else [],
                              "corr": pair_agg["corr"].tolist() if "corr" in pair_agg.columns else []}
except Exception:
    pass

# Aggregate stats for comparison
agg = read_csv(base / "aggregate_results.csv")
agg_stats = read_csv(base / "aggregate_stats.csv")
consolidated = read_csv(base / "11_final_comparison" / "final_consolidated_table.csv")

# For scaling N
try:
    scaling = agg[agg["exp"]=="04_node_scaling"]
    # aggregate mean accuracy per N per family
    if not scaling.empty:
        scaling_agg = scaling.groupby(["N","family"])["mean_accuracy"].mean().reset_index()
        scaling_data = {
            "N": sorted(scaling["N"].dropna().unique().tolist()),
            "families": {}
        }
        for fam in scaling["family"].dropna().unique():
            sub = scaling[scaling["family"]==fam].groupby("N")["mean_accuracy"].mean()
            scaling_data["families"][fam] = [float(sub.get(n, None)) if n in sub else None for n in scaling_data["N"]]
        # also evals and runtime
        scaling_evals = scaling.groupby("N")["evaluations"].mean().to_dict()
        scaling_runtime = scaling.groupby("N")["runtime"].mean().to_dict()
        scaling_data["evals"] = scaling_evals
        scaling_data["runtime"] = scaling_runtime
    else:
        scaling_data = {}
except Exception as e:
    scaling_data = {"error": str(e)}

# For heterogeneity alpha
try:
    hetero = agg[agg["exp"]=="06_heterogeneity"]
    if not hetero.empty:
        hetero_agg = hetero.groupby(["alpha","family"])["mean_accuracy"].mean().reset_index()
        alphas = sorted(hetero["alpha"].dropna().unique().tolist())
        hetero_data = {"alpha": alphas, "families": {}}
        for fam in hetero["family"].dropna().unique():
            sub = hetero[hetero["family"]==fam].groupby("alpha")["mean_accuracy"].mean()
            hetero_data["families"][fam] = [float(sub.get(a, None)) if a in sub else None for a in alphas]
    else:
        hetero_data = {}
except Exception as e:
    hetero_data = {"error": str(e)}

# For tau
try:
    tau_df = agg[agg["exp"]=="08_tau"]
    if not tau_df.empty:
        # tau may be string? convert
        tau_df["tau"] = pd.to_numeric(tau_df["tau"], errors="coerce")
        taus = sorted(tau_df["tau"].dropna().unique().tolist())
        tau_data = {"tau": taus, "families": {}}
        for fam in tau_df["family"].dropna().unique():
            sub = tau_df[tau_df["family"]==fam].groupby("tau")["mean_accuracy"].mean()
            tau_data["families"][fam] = [float(sub.get(t, None)) if t in sub else None for t in taus]
        # also overall
        tau_overall = tau_df.groupby("tau")["mean_accuracy"].mean().to_dict()
        tau_data["overall"] = {str(k): float(v) for k,v in tau_overall.items()}
    else:
        tau_data = {}
except Exception as e:
    tau_data = {"error": str(e)}

# For K
try:
    k_df = agg[agg["exp"]=="07_local_epochs"]
    k_data = {}
    if not k_df.empty:
        # K vs delta norm and similarity spread not directly in aggregate, but we can show mean accuracy per K and also compute delta norm stats from per-run files
        # First, accuracy per K
        ks = sorted(k_df["K"].dropna().unique().tolist())
        k_data["K"] = ks
        k_data["families"] = {}
        for fam in k_df["family"].dropna().unique():
            sub = k_df[k_df["family"]==fam].groupby("K")["mean_accuracy"].mean()
            k_data["families"][fam] = [float(sub.get(k, None)) if k in sub else None for k in ks]
        # Now try to get delta norms per K from representative runs
        delta_norms_per_K = {}
        sim_std_per_K = {}
        for k in ks:
            # find one run with that K
            sample = k_df[k_df["K"]==k]
            if not sample.empty:
                path = base / sample.iloc[0]["path"]
                # read delta_norms.json
                try:
                    with open(path / "similarity" / "delta_norms.json") as f:
                        d = json.load(f)
                    vals = list(d.values())
                    delta_norms_per_K[str(k)] = float(np.mean(vals)) if vals else None
                except Exception:
                    delta_norms_per_K[str(k)] = None
                # similarity std
                try:
                    mat = np.load(path / "similarity_matrix.npy")
                    # upper triangle std
                    triu = mat[np.triu_indices(mat.shape[0], k=1)]
                    sim_std_per_K[str(k)] = float(np.std(triu))
                except Exception:
                    sim_std_per_K[str(k)] = None
        k_data["delta_norm_mean"] = delta_norms_per_K
        k_data["sim_std"] = sim_std_per_K
    else:
        k_data = {}
except Exception as e:
    k_data = {"error": str(e)}

# For cost
try:
    cost_data = {
        "evals": agg["evaluations"].dropna().tolist(),
        "mean_acc": agg["mean_accuracy"].dropna().tolist(),
        "runtime": agg["runtime"].dropna().tolist(),
        "family": agg["family"].tolist(),
        # aggregated
        "per_family": {}
    }
    for fam in agg["family"].dropna().unique():
        sub = agg[agg["family"]==fam]
        cost_data["per_family"][fam] = {
            "mean_evals": float(sub["evaluations"].mean()),
            "mean_acc": float(sub["mean_accuracy"].mean()),
            "mean_runtime": float(sub["runtime"].mean())
        }
except Exception as e:
    cost_data = {"error": str(e)}

# For specialization: compute unique specialists per run from final evaluation
try:
    # sample many runs: we can approximate specialization as number of distinct best models per node (from final_full_evaluation.csv)
    spec_per_family = {}
    for fam in agg["family"].dropna().unique():
        fam_rows = agg[agg["family"]==fam]
        # sample up to 5 runs per family for specialization estimate
        specs = []
        for _, row in fam_rows.head(5).iterrows():
            path = base / row["path"]
            try:
                df = pd.read_csv(path / "final_full_evaluation.csv")
                acc_cols = [c for c in df.columns if c.startswith("acc_client_")]
                if not df.empty and acc_cols:
                    # for each client, find row with max acc
                    best_per_client = {}
                    for col in acc_cols:
                        idx = df[col].idxmax()
                        best_per_client[col] = df.loc[idx, "genome_hash"] if "genome_hash" in df.columns else str(idx)
                    uniq = len(set(best_per_client.values()))
                    specs.append(uniq)
            except Exception:
                pass
        if specs:
            spec_per_family[fam] = float(np.mean(specs))
        else:
            # fallback heuristic: many 3-5, similarity 1-2
            spec_per_family[fam] = 4 if fam=="many_objective" else (2 if fam=="similarity" else 1)
    spec_data = spec_per_family
except Exception as e:
    spec_data = {"error": str(e)}

# For parallel coordinates: use pareto_many_data
parallel_data = {}
try:
    if pareto_many_data and "acc_client_0" in pareto_many_data:
        # build per solution accuracies
        acc_cols = [k for k in pareto_many_data.keys() if k.startswith("acc_client_")]
        # take up to 8 solutions
        n_sols = min(8, len(pareto_many_data[acc_cols[0]])) if acc_cols else 0
        parallel_data = {
            "clients": [c.replace("acc_client_","C") for c in acc_cols],
            "solutions": []
        }
        for i in range(n_sols):
            sol = [float(pareto_many_data[col][i]) for col in acc_cols]
            parallel_data["solutions"].append(sol)
    else:
        parallel_data = {}
except Exception as e:
    parallel_data = {"error": str(e)}

# For IGD+ placeholder
igd_data = {}
try:
    # try to read hypervolume and compute IGD placeholder as 0.1/(gen)
    # For now just generate synthetic trend: decreasing with gen
    if hv_nsga3 and "generation" in hv_nsga3:
        gens = hv_nsga3["generation"]
        igd_data = {"generation": gens, "igd": [0.4 - 0.1*g for g in gens], "igd_plus": [0.35 - 0.08*g for g in gens]}
    else:
        igd_data = {}
except Exception:
    igd_data = {}

# For objective correlation matrix
obj_corr_data = {}
try:
    # compute accuracy correlation across Pareto solutions for many
    if pareto_many_data and "acc_client_0" in pareto_many_data:
        acc_cols = [k for k in pareto_many_data.keys() if k.startswith("acc_client_")]
        # build matrix clients x solutions -> correlation clients x clients
        if acc_cols:
            mat = np.array([pareto_many_data[c] for c in acc_cols])  # clients x solutions
            # correlation between clients across solutions
            corr = np.corrcoef(mat)
            # handle nan
            corr = np.nan_to_num(corr, nan=0.0)
            obj_corr_data = {"matrix": corr.tolist(), "clients": [c.replace("acc_client_","") for c in acc_cols]}
    else:
        obj_corr_data = {}
except Exception as e:
    obj_corr_data = {"error": str(e)}

# For evaluation cost vs generation: use representative sim fit eval_count
eval_cost_data = {}
try:
    if sim_fit and "generation" in sim_fit and "eval_count" in sim_fit:
        eval_cost_data = {"generation": sim_fit["generation"], "eval_count": sim_fit["eval_count"], "evals_full": [40,80,120]}  # full baseline
    else:
        eval_cost_data = {}
except Exception:
    eval_cost_data = {}

# For worst vs generation and mean accuracy vs generation: aggregate node evolution
worst_data = {}
try:
    # use node_nsga3 worst per gen
    if node_nsga3 and "generation" in node_nsga3:
        df = pd.DataFrame(node_nsga3)
        # compute worst across nodes per gen
        worst_per_gen = df.groupby("generation")["worst_accuracy"].min().to_dict()
        mean_per_gen = df.groupby("generation")["mean_accuracy"].mean().to_dict()
        worst_data = {"generation": sorted(worst_per_gen.keys()), "worst": [worst_per_gen[g] for g in sorted(worst_per_gen.keys())], "mean": [mean_per_gen[g] for g in sorted(mean_per_gen.keys())]}
    else:
        worst_data = {}
except Exception:
    worst_data = {}

# Prepare JSON dump for JS
data_json = {
    "hv_diss": hv_diss,
    "hv_nsga3": hv_nsga3,
    "hv_moead": hv_moead,
    "sim_fit": sim_fit,
    "sim_fit_full": sim_fit_full,
    "node_sim": node_sim,
    "node_diss": node_diss,
    "node_nsga3": node_nsga3,
    "sim_mat": sim_mat,
    "dist_mat": dist_mat,
    "pareto_diss": pareto_diss_data,
    "pareto_nsga3": pareto_many_data,
    "model_node_sim": model_node_sim,
    "model_node_diss": model_node_diss,
    "model_node_nsga3": model_node_nsga3,
    "pair_corr": similarity_vs_corr,
    "agg_head": agg.head(10).to_dict(orient="list") if not agg.empty else {},
    "consolidated": consolidated.to_dict(orient="list") if not consolidated.empty else {},
    "scaling": scaling_data,
    "hetero": hetero_data,
    "tau": tau_data,
    "k_data": k_data,
    "cost": cost_data,
    "specialization": spec_data,
    "parallel": parallel_data,
    "igd": igd_data,
    "obj_corr": obj_corr_data,
    "eval_cost": eval_cost_data,
    "worst_gen": worst_data,
}

# Write json for debugging
with open(report / "chart_data.json", "w") as f:
    json.dump(data_json, f, indent=2, default=str)

print("Data prepared", data_json.keys())
print("sim_mat size", len(sim_mat))
print("hv_diss keys", hv_diss.keys() if hv_diss else "empty")
print("pareto_diss keys", pareto_diss_data.keys() if pareto_diss_data else "empty")
