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
except Exception:
    scaling = pd.DataFrame()

# For heterogeneity alpha
try:
    hetero = agg[agg["exp"]=="06_heterogeneity"]
except Exception:
    hetero=pd.DataFrame()

# For tau
try:
    tau_df = agg[agg["exp"]=="08_tau"]
except Exception:
    tau_df=pd.DataFrame()

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
}

# Write json for debugging
with open(report / "chart_data.json", "w") as f:
    json.dump(data_json, f, indent=2, default=str)

print("Data prepared", data_json.keys())
print("sim_mat size", len(sim_mat))
print("hv_diss keys", hv_diss.keys() if hv_diss else "empty")
print("pareto_diss keys", pareto_diss_data.keys() if pareto_diss_data else "empty")
