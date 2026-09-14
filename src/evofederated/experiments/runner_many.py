"""Runner many-objective + PCA antes de cosine similarity."""
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
import dataclasses
import platform

import numpy as np
import pandas as pd
import torch

from ..utils.seed import set_seed
from ..utils.config import ExperimentConfig
from ..data.datasets import (
    make_synthetic_dataset,
    get_dataset,
    dirichlet_partition,
    heterogeneity_metrics,
    get_input_dim_and_n_classes,
)
from ..clients.client import FederatedClient
from ..federated.characterization import characterize_clients, flatten_params
from ..federated.similarity import compute_similarity_matrix, compute_distance_matrix_from_similarity, similarity_stats
from ..federated.pca import apply_pca, compute_similarity_preservation
from ..federated.matching import max_similarity_matching, matching_summary
from ..evolution.many_objective import ManyObjectiveEvaluator
from ..evolution.moead_runner import run_moead
from ..evolution.nsga3_runner import run_nsga3
from ..metrics.many_objective import (
    compute_hv, compute_igd, get_nondominated, coverage_metric, find_knee_points, objective_statistics
)
from ..models.genome import Genome


def _device(train_device: str):
    if train_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return train_device


def run_many_objective_experiment(
    cfg: ExperimentConfig,
    optimizer: str = "moead",  # moead | nsga3
    pca_mode: str = "none",  # none | variance | fixed
    pca_variance: float = 0.95,
    pca_components: int = 4,
    output_dir: Path = None,
    tau: float = None,
    verbose: bool = False,
):
    """
    Pipeline:
      dataset -> partition -> clients
      -> characterization Δ_i
      -> PCA (raw vs pca)
      -> similarity matrices S_raw, S_pca, preservation
      -> optimizer many-objective (N objectives)
    optimizer y PCA son factores independientes (factorial).
    Los objetivos son N silos = N objetivos, maximize accuracy_i => minimize 1-acc_i (explícito).
    Pairing se registra como información experimental pero NO reduce objetivos en modo full.
    """
    t_start = time.time()
    set_seed(cfg.seed)
    optimizer = optimizer.lower()
    assert optimizer in ("moead", "nsga3", "nsga-iii")

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(cfg.output_dir) / f"many_{optimizer}_{pca_mode}_{ts}"
    output_dir = Path(output_dir)
    for sub in ["dataset", "characterization", "similarity", "similarity_raw", "similarity_pca", "generations", "plots", "final_evaluation", "pareto"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # save config
    from ..utils.config import save_config
    save_config(cfg, output_dir / "config.json")
    try:
        save_config(cfg, output_dir / "config.yaml")
    except Exception:
        pass
    # save run meta
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump({
            "optimizer": optimizer,
            "pca_mode": pca_mode,
            "pca_variance": pca_variance,
            "pca_components": pca_components,
            "tau": tau,
            "seed": cfg.seed,
            "n_clients": cfg.dataset.n_clients,
            "K": cfg.characterization.k_epochs,
            "alpha": cfg.dataset.alpha,
            "pop_size": cfg.evolution.pop_size,
            "n_generations": cfg.evolution.n_generations,
        }, f, indent=2)

    device = _device(cfg.train.device)

    # Dataset
    t_data = time.time()
    if cfg.dataset.name == "synthetic":
        base_dataset = make_synthetic_dataset(
            n_samples=cfg.dataset.synthetic_n_samples,
            n_features=cfg.dataset.synthetic_n_features,
            n_informative=cfg.dataset.synthetic_n_informative,
            n_classes=cfg.dataset.synthetic_n_classes,
            seed=cfg.seed,
        )
        labels = base_dataset.tensors[1].numpy()
    else:
        train_set = get_dataset(cfg.dataset.name, root=cfg.dataset.data_root, train=True, download=True)
        if hasattr(train_set, "targets"):
            labels = np.array(train_set.targets) if not isinstance(train_set.targets, torch.Tensor) else train_set.targets.numpy()
        else:
            labels = np.array([train_set[i][1] for i in range(len(train_set))])
        base_dataset = train_set

    input_dim, n_classes = get_input_dim_and_n_classes(base_dataset)
    partition = dirichlet_partition(
        labels=labels,
        n_clients=cfg.dataset.n_clients,
        alpha=cfg.dataset.alpha,
        seed=cfg.dataset.seed,
        min_samples_per_client=cfg.dataset.min_samples_per_client,
        val_ratio=cfg.dataset.val_ratio,
        test_ratio=cfg.dataset.test_ratio,
    )
    clients = {}
    for cid in range(cfg.dataset.n_clients):
        from torch.utils.data import Subset
        train_idxs = partition.splits[cid]["train"]
        val_idxs = partition.splits[cid]["val"]
        test_idxs = partition.splits[cid]["test"]
        train_ds = Subset(base_dataset, train_idxs)
        val_ds = Subset(base_dataset, val_idxs) if len(val_idxs)>0 else None
        test_ds = Subset(base_dataset, test_idxs) if len(test_idxs)>0 else None
        clients[cid] = FederatedClient(cid, train_ds, val_ds, test_ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)

    het = heterogeneity_metrics(partition.class_distribution)
    with open(output_dir / "dataset" / "heterogeneity.json", "w") as f:
        json.dump({"partition": partition.to_dict(), "heterogeneity": het, "input_dim": input_dim, "n_classes": n_classes}, f, indent=2, default=str)
    pd.DataFrame(partition.class_distribution, columns=[f"class_{i}" for i in range(partition.n_classes)]).to_csv(output_dir / "dataset" / "class_distribution.csv", index_label="client")

    data_time = time.time() - t_data

    # Characterization Δ_i
    t_char = time.time()
    char_res = characterize_clients(
        clients=clients,
        decoder_input_dim=input_dim,
        decoder_n_classes=n_classes,
        genome_config=dataclasses.asdict(cfg.genome),
        train_config=dataclasses.asdict(cfg.train),
        k_epochs=cfg.characterization.k_epochs,
        seed=cfg.characterization.seed,
        device=device,
    )
    char_time = time.time() - t_char

    # Similarity RAW
    t_sim = time.time()
    S_raw = compute_similarity_matrix(char_res["deltas"])
    D_raw = compute_distance_matrix_from_similarity(S_raw)
    stats_raw = similarity_stats(S_raw)
    np.save(output_dir / "similarity_raw" / "similarity_matrix.npy", S_raw)
    np.save(output_dir / "similarity_raw" / "distance_matrix.npy", D_raw)
    pd.DataFrame(S_raw).to_csv(output_dir / "similarity_raw" / "similarity_matrix.csv", index=False)
    pd.DataFrame(D_raw).to_csv(output_dir / "similarity_raw" / "distance_matrix.csv", index=False)
    # also compatibility path
    np.save(output_dir / "similarity" / "similarity_raw.npy", S_raw)
    # pairing raw
    pairs_raw, sing_raw = max_similarity_matching(S_raw)
    summary_raw = matching_summary(S_raw, pairs_raw, sing_raw)

    # PCA pipeline
    # apply PCA to deltas before similarity
    deltas_pca, pca_info = apply_pca(
        char_res["deltas"],
        mode=pca_mode,
        variance=pca_variance,
        n_components=pca_components,
        seed=cfg.characterization.seed,
    )
    S_pca = compute_similarity_matrix(deltas_pca)
    D_pca = compute_distance_matrix_from_similarity(S_pca)
    stats_pca = similarity_stats(S_pca)
    np.save(output_dir / "similarity_pca" / "similarity_matrix.npy", S_pca)
    np.save(output_dir / "similarity_pca" / "distance_matrix.npy", D_pca)
    pd.DataFrame(S_pca).to_csv(output_dir / "similarity_pca" / "similarity_matrix.csv", index=False)
    # Save pca info
    with open(output_dir / "similarity_pca" / "pca_info.json", "w") as f:
        json.dump(pca_info, f, indent=2)
    # also store explained variance per component csv
    if pca_info.get("explained_variance_ratio"):
        pd.DataFrame({"component": range(1, len(pca_info["explained_variance_ratio"])+1),
                      "explained_variance_ratio": pca_info["explained_variance_ratio"],
                      "cumulative": np.cumsum(pca_info["explained_variance_ratio"]).tolist()
                      }).to_csv(output_dir / "similarity_pca" / "pca_explained_variance.csv", index=False)

    # If pca_mode == none, S_pca == S_raw, preservation trivial
    if pca_mode == "none":
        preservation = {"pearson": 1.0, "spearman": 1.0, "kendall": 1.0, "mae": 0.0, "rmse": 0.0, "nn_preserved": 1.0, "pairing_preserved": 1.0}
        pairs_pca, sing_pca = pairs_raw, sing_raw
        summary_pca = summary_raw
    else:
        preservation = compute_similarity_preservation(S_raw, S_pca)
        pairs_pca, sing_pca = max_similarity_matching(S_pca)
        summary_pca = matching_summary(S_pca, pairs_pca, sing_pca)

    # Save pairing and preservation
    with open(output_dir / "similarity" / "similarity_stats.json", "w") as f:
        json.dump({
            "raw": stats_raw,
            "pca": stats_pca,
            "pca_info": pca_info,
            "preservation": preservation,
            "pairs_raw": summary_raw,
            "pairs_pca": summary_pca,
            "computation_time": time.time() - t_sim,
        }, f, indent=2)
    # Save pairings individually
    client_ids_sorted = sorted(char_res["deltas"].keys())
    # For raw vs pca we save pairs as csv
    for suffix, pairs, summ, S in [("raw", pairs_raw, summary_raw, S_raw), ("pca", pairs_pca, summary_pca, S_pca)]:
        rows = []
        for idx, (a,b) in enumerate(pairs):
            rows.append({"pair_id": idx, "pos_a": int(a), "pos_b": int(b), "client_a": int(client_ids_sorted[a]), "client_b": int(client_ids_sorted[b]), "similarity": float(S[a,b])})
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / "similarity" / f"pairs_{suffix}.csv", index=False)
        with open(output_dir / "similarity" / f"pairs_{suffix}.json", "w") as f:
            json.dump(summ, f, indent=2)

    # Also probabilistic pairing if tau given (for experiment B)
    prob_info = {}
    if tau is not None:
        from ..federated.matching import probabilistic_similarity_pairing
        rng = np.random.RandomState(cfg.seed + 202)
        # For both raw and pca, store probabilistic pairing stats
        for suffix, S in [("raw", S_raw), ("pca", S_pca)]:
            pairs_prob, sing_prob, info_prob = probabilistic_similarity_pairing(S, tau, rng)
            prob_info[suffix] = {"pairs": pairs_prob, "singletons": sing_prob, "info": info_prob}
        with open(output_dir / "similarity" / "probabilistic_pairing.json", "w") as f:
            json.dump(prob_info, f, indent=2, default=str)

    # True TV for correlation offline (no leakage)
    try:
        props = partition.class_distribution / partition.class_distribution.sum(axis=1, keepdims=True).clip(min=1)
        n = cfg.dataset.n_clients
        true_TV = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                true_TV[i,j] = np.abs(props[i]-props[j]).sum()/2
        np.save(output_dir / "similarity" / "true_TV_matrix.npy", true_TV)
    except Exception:
        pass

    t_sim_total = time.time() - t_sim

    # Many-objective evolution
    l_max = cfg.genome.l_max
    n_var = 1 + l_max + 3
    evaluator = ManyObjectiveEvaluator(
        clients=clients,
        input_dim=input_dim,
        n_classes=n_classes,
        genome_config=dataclasses.asdict(cfg.genome),
        train_config=dataclasses.asdict(cfg.train),
        device=device,
        seed=cfg.evolution.seed,
        generation=0,
    )
    t_evo = time.time()
    if optimizer in ("moead",):
        evo_res = run_moead(
            evaluator=evaluator,
            n_var=n_var,
            pop_size=cfg.evolution.pop_size,
            n_generations=cfg.evolution.n_generations,
            decomposition="tchebycheff",
            n_neighbors=min(20, cfg.evolution.pop_size),
            seed=cfg.evolution.seed,
            verbose=verbose,
        )
    else:
        evo_res = run_nsga3(
            evaluator=evaluator,
            n_var=n_var,
            pop_size=cfg.evolution.pop_size,
            n_generations=cfg.evolution.n_generations,
            seed=cfg.evolution.seed,
            verbose=verbose,
        )
    evo_time = time.time() - t_evo

    # Save histories
    hv_hist = evo_res["hv_history"]
    pd.DataFrame(hv_hist).to_csv(output_dir / "generations" / "hypervolume.csv", index=False)
    pd.DataFrame(hv_hist).to_csv(output_dir / "hypervolume.csv", index=False)
    # also generation_metrics concatenated
    if hv_hist:
        df_hv = pd.DataFrame(hv_hist)
        # compute generation metrics csv with hypervolume, nondominated, mean/worst
        df_hv.to_csv(output_dir / "generations" / "generation_metrics.csv", index=False)
        df_hv.to_csv(output_dir / "generation_metrics.csv", index=False)

    # Save per-individual objective values
    df_ind = pd.DataFrame(evaluator.generation_history)
    if not df_ind.empty:
        df_ind.to_csv(output_dir / "generations" / "population_history.csv", index=False)
        df_ind.to_csv(output_dir / "objective_values.csv", index=False)
        df_ind.to_csv(output_dir / "per_client_metrics.csv", index=False)
        # per_generation aggregated also in objective_values
    else:
        pd.DataFrame().to_csv(output_dir / "objective_values.csv", index=False)

    # Pareto front of final population
    res = evo_res["result"]
    F_final = res.pop.get("F") if res.pop is not None else np.array([])
    X_final = res.pop.get("X") if res.pop is not None else np.array([])
    hv_ref = evo_res["hv_ref"]
    # get nondominated
    if F_final is not None and len(F_final)>0:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds = NonDominatedSorting().do(F_final, only_non_dominated_front=True)
        is_nd = np.zeros(len(F_final), dtype=bool)
        is_nd[nds] = True
        pf_F = F_final[nds] if len(nds)>0 else F_final
        # build pareto dataframe
        pareto_rows = []
        for rank, idx in enumerate(nds):
            vec = X_final[idx]
            genome = Genome.from_vector(vec, dataclasses.asdict(cfg.genome))
            # retrieve accuracies from evaluator history for final gen individuals? We'll recompute from F (1-acc)
            accs = 1 - F_final[idx]
            f1s = accs  # placeholder
            row = {
                "pareto_idx": int(rank),
                "pop_idx": int(idx),
                "genome": json.dumps(genome.to_dict()),
                "genome_L": genome.L,
                "genome_hidden": str(genome.hidden_sizes),
                "genome_act": genome.activation,
                "genome_hash": hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()[:12],
                "mean_accuracy": float(np.mean(accs)),
                "worst_accuracy": float(np.min(accs)),
                "std_accuracy": float(np.std(accs)),
                "gap": float(np.max(accs)-np.min(accs)),
            }
            # per client accuracy and objectives
            for cid_idx, cid in enumerate(sorted(clients.keys())):
                row[f"acc_client_{cid}"] = float(accs[cid_idx])
                row[f"obj_{cid}"] = float(F_final[idx, cid_idx])
            pareto_rows.append(row)
        df_pareto = pd.DataFrame(pareto_rows)
        df_pareto.to_csv(output_dir / "pareto" / "pareto_front.csv", index=False)
        df_pareto.to_csv(output_dir / "pareto_front.csv", index=False)
        df_pareto.to_csv(output_dir / "non_dominated.csv", index=False)

        # knee points
        knee_indices = find_knee_points(F_final)
        knee_rows = []
        for k_idx in knee_indices:
            if k_idx < len(F_final):
                vec = X_final[k_idx]
                genome = Genome.from_vector(vec, dataclasses.asdict(cfg.genome))
                accs = 1 - F_final[k_idx]
                knee_rows.append({
                    "pop_idx": int(k_idx),
                    "genome": json.dumps(genome.to_dict()),
                    "mean_accuracy": float(np.mean(accs)),
                    "worst_accuracy": float(np.min(accs)),
                })
        pd.DataFrame(knee_rows).to_csv(output_dir / "pareto" / "knee_points.csv", index=False)

        # Representative models: best mean, best worst, knee, extremes (min per objective -> max accuracy per client)
        # extremes: for each client best accuracy among PF
        reps = {}
        # best mean
        if pareto_rows:
            df_tmp = pd.DataFrame(pareto_rows)
            best_mean_idx = int(df_tmp["mean_accuracy"].idxmax())
            reps["best_mean"] = pareto_rows[best_mean_idx]
            best_worst_idx = int(df_tmp["worst_accuracy"].idxmax())
            reps["best_worst"] = pareto_rows[best_worst_idx]
        # balanced (closest to ideal)
        if knee_rows:
            reps["knee"] = knee_rows[0]
        # extremes per client (max acc per client)
        extremes = []
        for cid_idx, cid in enumerate(sorted(clients.keys())):
            col = f"acc_client_{cid}"
            if pareto_rows and col in df_tmp.columns:
                idx = int(df_tmp[col].idxmax())
                extremes.append(pareto_rows[idx])
        # deduplicate extremes by hash
        seen = set()
        uniq_ext = []
        for r in extremes:
            h = r["genome_hash"]
            if h not in seen:
                seen.add(h)
                uniq_ext.append(r)
        reps["extremes"] = uniq_ext
        # save representative summary
        with open(output_dir / "pareto" / "representatives.json", "w") as f:
            # convert numpy to python
            json.dump(reps, f, indent=2, default=str)
        # also exhaustive final evaluation for representatives (already evaluated but we re-validate)
        # In full mode they are already exact, so we just copy per-client metrics from pareto rows
        # Save aggregated per-client metrics for representatives
        rep_rows = []
        for key in ["best_mean", "best_worst", "knee"]:
            if key in reps and isinstance(reps[key], dict) and "mean_accuracy" in reps[key]:
                r = reps[key]
                r_copy = dict(r)
                r_copy["type"] = key
                rep_rows.append(r_copy)
        for r in uniq_ext[:5]:  # top 5 extremes
            rc = dict(r)
            rc["type"] = "extreme"
            rep_rows.append(rc)
        if rep_rows:
            pd.DataFrame(rep_rows).to_csv(output_dir / "final_evaluation" / "representatives_metrics.csv", index=False)
            pd.DataFrame(rep_rows).to_csv(output_dir / "per_client_metrics.csv", index=False)

        # Hypervolume final
        from ..metrics.many_objective import compute_hv
        hv_final = compute_hv(F_final, hv_ref)
        hv_pf = compute_hv(pf_F, hv_ref)
        hv_info = {"hv_final_pop": float(hv_final), "hv_pareto": float(hv_pf), "hv_ref": hv_ref.tolist(), "n_pop": int(len(F_final)), "n_pareto": int(len(pf_F))}
    else:
        hv_info = {"hv_final_pop": 0.0, "hv_pareto": 0.0, "hv_ref": hv_ref.tolist() if "hv_ref" in locals() else [], "n_pop": 0, "n_pareto": 0}
        pd.DataFrame().to_csv(output_dir / "pareto_front.csv", index=False)

    with open(output_dir / "pareto" / "hypervolume.json", "w") as f:
        json.dump(hv_info, f, indent=2)

    # IGD if we have reference PF (combine across runs not available here, so compute self)
    # Use final PF as reference for per-generation IGD evolution (approximation)
    # We'll compute IGD of each generation's PF vs final PF
    try:
        from pymoo.indicators.igd import IGD
        if F_final is not None and len(F_final)>0 and "pf_F" in locals() and len(pf_F)>0:
            # per generation PFs from evaluator? Not stored, but we can compute from generation_history grouping
            igd_rows = []
            # approximate: each generation's nondominated from that generation's individuals
            if not df_ind.empty:
                for gen in sorted(df_ind["generation"].unique()):
                    sub = df_ind[df_ind["generation"]==gen]
                    # reconstruct F for that gen from objective columns obj_*
                    obj_cols = [c for c in sub.columns if c.startswith("obj_")]
                    if not obj_cols:
                        continue
                    F_gen = sub[obj_cols].values
                    from ..metrics.many_objective import compute_igd
                    igd_val = compute_igd(F_gen, pf_F)
                    igd_rows.append({"generation": int(gen), "igd": float(igd_val) if not np.isnan(igd_val) else 0.0})
                if igd_rows:
                    pd.DataFrame(igd_rows).to_csv(output_dir / "generations" / "igd.csv", index=False)
                    pd.DataFrame(igd_rows).to_csv(output_dir / "igd.csv", index=False)
    except Exception as e:
        print(f"IGD failed {e}")

    # Cost and runtime
    total_time = time.time() - t_start
    cost_info = {
        "total_time": float(total_time),
        "data_time": float(data_time),
        "char_time": float(char_time),
        "sim_time": float(t_sim_total),
        "evo_time": float(evo_time),
        "eval_count": int(evaluator.eval_count),
        "n_obj": int(evaluator.n_obj),
        "n_clients": int(cfg.dataset.n_clients),
        "pop_size_actual": int(evo_res.get("pop_size_actual", cfg.evolution.pop_size)),
        "n_generations": int(cfg.evolution.n_generations),
        "total_evals_possible_full": int(cfg.evolution.pop_size * cfg.evolution.n_generations * cfg.dataset.n_clients),
        "evaluations": int(evaluator.eval_count),
        "optimizer": optimizer,
        "pca_mode": pca_mode,
    }
    with open(output_dir / "runtime.json", "w") as f:
        json.dump(cost_info, f, indent=2)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            "hv": hv_info,
            "cost": cost_info,
            "heterogeneity": het,
            "pca": pca_info,
            "preservation": preservation,
        }, f, indent=2)

    # metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "optimizer": optimizer,
            "pca_mode": pca_mode,
            "pca_variance": pca_variance,
            "pca_components": pca_components,
            "dataset": dataclasses.asdict(cfg.dataset),
            "train": dataclasses.asdict(cfg.train),
            "genome": dataclasses.asdict(cfg.genome),
            "evolution": {**dataclasses.asdict(cfg.evolution), "pop_size_actual": cost_info["pop_size_actual"]},
            "hv_ref": hv_ref.tolist(),
            "ref_dirs_shape": evo_res["ref_dirs"].shape if "ref_dirs" in evo_res else [],
            "n_partitions": evo_res.get("n_partitions"),
            "ref_dirs_info": f"p={evo_res.get('n_partitions')}, actual_pop={cost_info['pop_size_actual']}",
            "device": device,
            "seed": cfg.seed,
            "total_time": total_time,
            "platform": platform.platform(),
            "python": platform.python_version(),
        }, f, indent=2, default=str)

    # summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "optimizer": optimizer,
            "pca_mode": pca_mode,
            "pca": pca_info,
            "preservation": preservation,
            "hv": hv_info,
            "cost": cost_info,
            "mean_accuracy_best": float(max([r["mean_accuracy"] for r in pareto_rows])) if 'pareto_rows' in locals() and pareto_rows else 0.0,
            "worst_accuracy_best": float(max([r["worst_accuracy"] for r in pareto_rows])) if 'pareto_rows' in locals() and pareto_rows else 0.0,
            "n_pareto": int(hv_info.get("n_pareto",0)),
        }, f, indent=2)

    return {
        "output_dir": output_dir,
        "hv_history": hv_hist,
        "evaluator": evaluator,
        "cost": cost_info,
        "hv_info": hv_info,
        "pca_info": pca_info,
    }
