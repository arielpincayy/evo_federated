"""Runner for similarity-pairing single-objective variant.

Pipeline:
  Dataset -> Dirichlet partition -> Clients
  -> Reference model w0 -> K epochs per silo -> Δ_i
  -> Cosine similarity matrix S -> max-weight matching -> pairs / singletons
  -> GA single-objective with alternating representatives
  -> Final full evaluation on all silos
"""
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
    get_dataset,
    make_synthetic_dataset,
    dirichlet_partition,
    heterogeneity_metrics,
    get_input_dim_and_n_classes,
)
from ..clients.client import FederatedClient
from ..federated.characterization import characterize_clients, flatten_params, set_model_params
from ..federated.similarity import compute_similarity_matrix, compute_distance_matrix_from_similarity, similarity_stats
from ..federated.matching import max_similarity_matching, get_representatives, matching_summary, intra_pair_similarities
from ..evolution.similarity_evaluator import SimilarityEvaluator
from ..evolution.single_objective import run_single_objective_ga
from ..models.genome import Genome
from ..evolution.validation import exhaustive_evaluate_genome


def _get_device(train_device: str):
    if train_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return train_device


def run_similarity_experiment(
    cfg: ExperimentConfig,
    output_dir: Path = None,
    verbose: bool = False,
    pairing_mode: str = "deterministic",
    pairing_tau: float = None,
    rep_mode: str = "alternating",
    full_mode: bool = False,
):
    t_start = time.time()
    set_seed(cfg.seed)

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(cfg.output_dir) / f"similarity_{ts}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (output_dir / "characterization").mkdir(parents=True, exist_ok=True)
    (output_dir / "similarity").mkdir(parents=True, exist_ok=True)
    (output_dir / "generations").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "final_evaluation").mkdir(parents=True, exist_ok=True)

    from ..utils.config import save_config
    save_config(cfg, output_dir / "config.json")
    try:
        save_config(cfg, output_dir / "config.yaml")
    except Exception:
        pass

    device = _get_device(cfg.train.device)

    # --- Dataset / Partition -------------------------------------------------
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
        elif hasattr(train_set, "labels"):
            labels = np.array(train_set.labels)
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

    # Create clients
    clients = {}
    for cid in range(cfg.dataset.n_clients):
        train_idxs = partition.splits[cid]["train"]
        val_idxs = partition.splits[cid]["val"]
        test_idxs = partition.splits[cid]["test"]
        from torch.utils.data import Subset
        train_ds = Subset(base_dataset, train_idxs)
        val_ds = Subset(base_dataset, val_idxs) if len(val_idxs) > 0 else None
        test_ds = Subset(base_dataset, test_idxs) if len(test_idxs) > 0 else None
        client = FederatedClient(
            client_id=cid,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )
        clients[cid] = client

    het_metrics = heterogeneity_metrics(partition.class_distribution)
    with open(output_dir / "dataset" / "heterogeneity.json", "w") as f:
        json.dump({
            "partition": partition.to_dict(),
            "heterogeneity": het_metrics,
            "input_dim": input_dim,
            "n_classes": n_classes,
        }, f, indent=2, default=str)
    pd.DataFrame(
        partition.class_distribution,
        columns=[f"class_{i}" for i in range(partition.n_classes)],
    ).to_csv(output_dir / "dataset" / "class_distribution.csv", index_label="client")
    with open(output_dir / "dataset" / "heterogeneity_stats.json", "w") as f:
        json.dump(het_metrics, f, indent=2)

    data_time = time.time() - t_data

    # --- Characterization: w0 -> K epochs -> Δ_i ---------------------------
    t_char_start = time.time()
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
    t_char = time.time() - t_char_start

    # Similarity matrix
    t_sim_start = time.time()
    S = compute_similarity_matrix(char_res["deltas"])
    D = compute_distance_matrix_from_similarity(S)
    sim_stats = similarity_stats(S)
    # also divergence stats for compatibility
    from ..federated.divergence import divergence_stats as div_stats_fn
    div_stats = div_stats_fn(D)

    # Save matrices
    np.save(output_dir / "similarity" / "similarity_matrix.npy", S)
    np.save(output_dir / "similarity" / "distance_matrix.npy", D)
    # also keep legacy path for compatibility with plots expecting divergence_matrix
    np.save(output_dir / "characterization" / "similarity_matrix.npy", S)
    np.save(output_dir / "characterization" / "divergence_matrix.npy", D)
    pd.DataFrame(S).to_csv(output_dir / "similarity" / "similarity_matrix.csv", index=False)
    pd.DataFrame(D).to_csv(output_dir / "similarity" / "distance_matrix.csv", index=False)
    pd.DataFrame(S).to_csv(output_dir / "characterization" / "similarity_matrix.csv", index=False)
    pd.DataFrame(D).to_csv(output_dir / "characterization" / "divergence_matrix.csv", index=False)

    with open(output_dir / "similarity" / "similarity_stats.json", "w") as f:
        json.dump({
            "similarity_stats": sim_stats,
            "distance_stats": div_stats,
            "probe_genome": char_res["probe_genome"].to_dict(),
            "n_params": char_res["n_params"],
            "k_epochs": cfg.characterization.k_epochs,
            "computation_time": time.time() - t_sim_start,
            "note": "Similarity computed from Δ_i only, not dataset metrics. Pairs built maximizing intra-pair similarity via max-weight matching.",
        }, f, indent=2, default=str)
    # delta norms
    delta_norms = {str(cid): float(np.linalg.norm(v)) for cid, v in char_res["deltas"].items()}
    with open(output_dir / "similarity" / "delta_norms.json", "w") as f:
        json.dump(delta_norms, f, indent=2)
    with open(output_dir / "characterization" / "similarity_stats.json", "w") as f:
        json.dump({"similarity_stats": sim_stats, "distance_stats": div_stats}, f, indent=2, default=str)

    # Matching — support deterministic / random / probabilistic / full
    pairing_rng = np.random.RandomState(cfg.seed + 101)
    prob_info = {}
    n_clients_tmp = int(S.shape[0])
    if full_mode:
        pairs, singletons = [], list(range(n_clients_tmp))
        summary = {"n_clients": int(n_clients_tmp), "n_pairs": 0, "n_singletons": int(n_clients_tmp), "pairs": [], "singletons": list(range(n_clients_tmp)), "note": "full_mode: all clients representatives", "pairing_mode": "full"}
    elif pairing_mode == "random":
        from ..federated.matching import random_pairing
        pairs, singletons = random_pairing(S, pairing_rng)
        summary = matching_summary(S, pairs, singletons)
        summary["pairing_mode"] = "random"
    elif pairing_mode == "probabilistic":
        from ..federated.matching import probabilistic_similarity_pairing
        tau = pairing_tau if pairing_tau is not None else 0.5
        pairs, singletons, prob_info = probabilistic_similarity_pairing(S, tau, pairing_rng)
        summary = matching_summary(S, pairs, singletons)
        summary["pairing_mode"] = "probabilistic"
        summary["tau"] = float(tau)
        summary.update(prob_info)
    else:
        pairs, singletons = max_similarity_matching(S)
        summary = matching_summary(S, pairs, singletons)
        summary["pairing_mode"] = "deterministic"
    # Save pairing info - both pos indices and client ids
    client_ids_sorted = sorted(clients.keys())
    # pairs are pos indices; translate to client ids for readability
    pairs_cids = [(client_ids_sorted[a], client_ids_sorted[b]) for a, b in pairs]
    singletons_cids = [client_ids_sorted[s] for s in singletons]
    with open(output_dir / "similarity" / "pairs.json", "w") as f:
        json.dump({
            "pairs_pos": pairs,
            "pairs_client_ids": pairs_cids,
            "singletons_pos": singletons,
            "singletons_client_ids": singletons_cids,
            "summary": summary,
            "pairing_mode": pairing_mode,
            "pairing_tau": pairing_tau,
            "rep_mode": rep_mode,
            "full_mode": full_mode,
            "singletons_note": "Singleton groups for odd N: representative is itself always (no alternation).",
            "alternating_note": "For each pair (A,B): generation even -> A, odd -> B, deterministic per generation. Random uses per-generation RNG.",
            "prob_info": prob_info,
        }, f, indent=2, default=str)
    # also save as csv for easy inspection
    pair_rows = []
    for idx, (a, b) in enumerate(pairs):
        pair_rows.append({
            "pair_id": idx,
            "client_a_pos": int(a),
            "client_b_pos": int(b),
            "client_a_id": int(client_ids_sorted[a]),
            "client_b_id": int(client_ids_sorted[b]),
            "similarity": float(S[a, b]),
        })
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(output_dir / "similarity" / "pairs.csv", index=False)
    else:
        pd.DataFrame(columns=["pair_id","client_a_pos","client_b_pos","client_a_id","client_b_id","similarity"]).to_csv(output_dir / "similarity" / "pairs.csv", index=False)

    # True TV correlation (offline analysis, not used for pairing)
    try:
        props = partition.class_distribution / partition.class_distribution.sum(axis=1, keepdims=True).clip(min=1)
        n = cfg.dataset.n_clients
        true_TV = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                true_TV[i, j] = np.abs(props[i] - props[j]).sum() / 2
        np.save(output_dir / "similarity" / "true_TV_matrix.npy", true_TV)
        np.save(output_dir / "characterization" / "true_TV_matrix.npy", true_TV)
        try:
            from scipy.stats import spearmanr, kendalltau
            idx = np.triu_indices(n, k=1)
            spearman_corr, _ = spearmanr(S[idx], 1 - true_TV[idx])  # similarity vs (1-TV) should be positive if similar clients have similar updates
            # Instead we compare distance vs TV like before: D vs TV
            spearman_D, _ = spearmanr(D[idx], true_TV[idx])
            kendall_D, _ = kendalltau(D[idx], true_TV[idx])
            pearson_D = float(np.corrcoef(D[idx], true_TV[idx])[0,1]) if len(D[idx])>1 else 0.0
            # also similarity vs (1 - TV)
            pearson_S = float(np.corrcoef(S[idx], 1 - true_TV[idx])[0,1]) if len(S[idx])>1 else 0.0
        except Exception:
            spearman_D = kendall_D = pearson_D = pearson_S = spearman_corr = 0.0
        with open(output_dir / "similarity" / "proxy_validation.json", "w") as f:
            json.dump({
                "spearman_distance_vs_TV": float(spearman_D) if spearman_D is not None else None,
                "kendall_distance_vs_TV": float(kendall_D) if kendall_D is not None else None,
                "pearson_distance_vs_TV": float(pearson_D),
                "pearson_similarity_vs_invTV": float(pearson_S),
                "note": "Correlation between Delta-based similarity/distance and true TV. Offline only, not used for pairing.",
            }, f, indent=2)
    except Exception as e:
        print(f"[WARN] proxy validation failed: {e}")

    t_sim_total = time.time() - t_sim_start

    # --- Evolution single-objective -----------------------------------------
    l_max = cfg.genome.l_max
    n_var = 1 + l_max + 3
    client_ids_sorted = sorted(clients.keys())

    evaluator = SimilarityEvaluator(
        clients=clients,
        input_dim=input_dim,
        n_classes=n_classes,
        genome_config=dataclasses.asdict(cfg.genome),
        train_config=dataclasses.asdict(cfg.train),
        pairs=pairs,
        singletons=singletons,
        client_ids_sorted=client_ids_sorted,
        similarity_matrix=S,
        device=device,
        seed=cfg.evolution.seed,
        generation=0,
        rep_mode=rep_mode,
        full_mode=full_mode,
    )

    t_evo_start = time.time()
    ga_res = run_single_objective_ga(
        evaluator=evaluator,
        n_var=n_var,
        pop_size=cfg.evolution.pop_size,
        n_generations=cfg.evolution.n_generations,
        crossover_prob=cfg.evolution.crossover_prob,
        crossover_eta=cfg.evolution.crossover_eta,
        mutation_prob=cfg.evolution.mutation_prob,
        mutation_eta=cfg.evolution.mutation_eta,
        seed=cfg.evolution.seed,
        verbose=verbose,
    )
    t_evo = time.time() - t_evo_start

    # Save evolution histories
    # per-generation aggregated history from tracker
    df_gen = pd.DataFrame(ga_res["history"])
    if not df_gen.empty:
        df_gen.to_csv(output_dir / "generations" / "generation_metrics.csv", index=False)
        df_gen.to_csv(output_dir / "generation_metrics.csv", index=False)
    # per-individual history from evaluator
    df_ind = pd.DataFrame(evaluator.generation_history)
    if not df_ind.empty:
        df_ind.to_csv(output_dir / "generations" / "population_history.csv", index=False)
        df_ind.to_csv(output_dir / "population_history.csv", index=False)
        df_ind.to_csv(output_dir / "evaluations.csv", index=False)
    # also save hyper-like history as fitness_history.csv
    if not df_gen.empty:
        df_gen[["generation","best_fitness","mean_fitness","median_fitness","std_fitness","worst_fitness","eval_count","elapsed"]].to_csv(output_dir / "fitness_history.csv", index=False)

    # Determine best genome
    if not df_ind.empty:
        best_row = df_ind.loc[df_ind["mean_accuracy"].idxmax()]
        try:
            best_genome_dict = json.loads(best_row["genome"])
            best_genome = Genome.from_dict(best_genome_dict)
        except Exception:
            # fallback reconstruct from columns
            best_genome = Genome.random(dataclasses.asdict(cfg.genome), np.random.RandomState(cfg.seed))
            # but try vector approach
        best_fitness = float(best_row["mean_accuracy"])
        best_gen = int(best_row["generation"])
    else:
        # fallback: use GA result.pop best
        res = ga_res["result"]
        if res.pop is not None and len(res.pop.get("F")):
            # F is 1 - accuracy, smallest is best
            F = res.pop.get("F")
            X = res.pop.get("X")
            best_idx = int(np.argmin(F))
            best_vec = X[best_idx]
            best_genome = Genome.from_vector(best_vec, dataclasses.asdict(cfg.genome))
            best_fitness = float(1 - F[best_idx,0])
            best_gen = cfg.evolution.n_generations
        else:
            best_genome = None
            best_fitness = 0.0
            best_gen = 0

    # Keep top-k genomes for optional evaluation (k=3)
    top_genomes = []
    if not df_ind.empty:
        df_sorted = df_ind.sort_values("mean_accuracy", ascending=False).drop_duplicates(subset=["genome_hash"])
        # top-3 unique
        for _, r in df_sorted.head(3).iterrows():
            try:
                g = Genome.from_dict(json.loads(r["genome"]))
                top_genomes.append((g, float(r["mean_accuracy"]), r["genome_hash"]))
            except Exception:
                continue
    if not top_genomes and best_genome is not None:
        top_genomes = [(best_genome, best_fitness, "best")]

    # --- Final full evaluation ------------------------------------------------
    t_final_start = time.time()
    final_rows = []
    # Evaluate best and optionally top-k full on all silos
    # Use exhaustive_evaluate_genome for detailed metrics
    from ..evolution.validation import exhaustive_evaluate_genome as exh_eval

    # For best only we compute detailed per-client metrics plus aggregated
    # We'll also evaluate all top_k for completeness
    all_final = []
    for rank, (genome, fit, gh) in enumerate(top_genomes):
        ev = exh_eval(genome, clients, input_dim, n_classes, dataclasses.asdict(cfg.train), device=device)
        # ev contains f1s, accs, losses, mean etc.
        row = {
            "rank": int(rank),
            "genome_hash": gh,
            "genome": str(genome.to_dict()),
            "genome_L": genome.L,
            "genome_hidden": str(genome.hidden_sizes),
            "genome_act": genome.activation,
            "genome_dropout": genome.dropout,
            "genome_bn": genome.use_bn,
            "mean_accuracy": float(ev["mean_acc"]),
            "median_accuracy": float(np.median(ev["accs"])) if ev["accs"] else 0.0,
            "min_accuracy": float(ev["min_acc"]),
            "max_accuracy": float(np.max(ev["accs"])) if ev["accs"] else 0.0,
            "std_accuracy": float(ev["std_acc"]),
            "var_accuracy": float(np.var(ev["accs"])) if ev["accs"] else 0.0,
            "gap_max_min": float(np.max(ev["accs"]) - np.min(ev["accs"])) if ev["accs"] else 0.0,
            "worst_client_accuracy": float(ev["min_acc"]),
            "best_client_accuracy": float(np.max(ev["accs"])) if ev["accs"] else 0.0,
            "mean_f1": float(ev["mean_f1"]),
            "median_f1": float(ev["median_f1"]),
            "min_f1": float(ev["min_f1"]),
            "max_f1": float(ev["max_f1"]),
            "std_f1": float(ev["std_f1"]),
            "mean_loss": float(np.mean(ev["losses"])) if ev["losses"] else 0.0,
            "evolution_fitness": float(fit),
            "evolution_generation": int(best_gen) if rank==0 else -1,
        }
        # percentiles
        if ev["accs"]:
            acc_arr = np.array(ev["accs"])
            row["p25_accuracy"] = float(np.percentile(acc_arr, 25))
            row["p75_accuracy"] = float(np.percentile(acc_arr, 75))
            row["p10_accuracy"] = float(np.percentile(acc_arr, 10))
            row["p90_accuracy"] = float(np.percentile(acc_arr, 90))
        # macro average (mean accuracy already)
        row["macro_avg_accuracy"] = row["mean_accuracy"]
        for cid_idx, cid in enumerate(sorted(clients.keys())):
            row[f"acc_client_{cid}"] = float(ev["accs"][cid_idx]) if cid_idx < len(ev["accs"]) else 0.0
            row[f"f1_client_{cid}"] = float(ev["f1s"][cid_idx]) if cid_idx < len(ev["f1s"]) else 0.0
            row[f"loss_client_{cid}"] = float(ev["losses"][cid_idx]) if cid_idx < len(ev["losses"]) else 0.0
        all_final.append(row)
        if rank == 0:
            final_rows = [row]

    df_final = pd.DataFrame(all_final)
    if not df_final.empty:
        df_final.to_csv(output_dir / "final_evaluation" / "per_client_metrics.csv", index=False)
        df_final.to_csv(output_dir / "final_evaluation.csv", index=False)
        # also best only
        df_best = df_final.head(1)
        df_best.to_csv(output_dir / "final_evaluation" / "best_full_evaluation.csv", index=False)
        df_best.to_csv(output_dir / "best_full_evaluation.csv", index=False)

    t_final = time.time() - t_final_start
    total_elapsed = time.time() - t_start

    # --- Cost metrics --------------------------------------------------------
    n_reps = len(pairs) + len(singletons)  # per generation representatives count
    # e.g., for N=8 pairs 4 => reps 4 per gen
    evals_per_gen = cfg.evolution.pop_size * n_reps
    total_search_evals = int(evaluator.eval_count)  # actual
    theoretical_full_evals = cfg.evolution.pop_size * cfg.dataset.n_clients * cfg.evolution.n_generations
    # characterization evaluations: N * K? But our char cost counts as N evaluations (one per client K epochs)
    char_evals = cfg.dataset.n_clients  # number of Δ computations (each K epochs)
    # Actually each Δ required K epochs training, count as char_evals
    evals_avoided = theoretical_full_evals - total_search_evals
    reduction_percent = (1 - total_search_evals / theoretical_full_evals) * 100 if theoretical_full_evals else 0.0
    # Final full evaluation cost: N * top_k (here 3 or 1)
    final_full_evals = len(all_final) * cfg.dataset.n_clients
    total_with_final = total_search_evals + char_evals + final_full_evals

    cost = {
        "n_clients": int(cfg.dataset.n_clients),
        "n_pairs": int(len(pairs)),
        "n_singletons": int(len(singletons)),
        "n_representatives_per_gen": int(n_reps),
        "pop_size": int(cfg.evolution.pop_size),
        "n_generations": int(cfg.evolution.n_generations),
        "total_search_evaluations": int(total_search_evals),
        "theoretical_full_evaluations": int(theoretical_full_evals),
        "evaluations_per_generation": int(evals_per_gen),
        "evals_avoided_vs_full": int(evals_avoided),
        "reduction_percent_vs_full": float(reduction_percent),
        "characterization_evaluations": int(char_evals),
        "characterization_k_epochs": int(cfg.characterization.k_epochs),
        "characterization_time_sec": float(t_char),
        "similarity_matrix_time_sec": float(t_sim_total),
        "evolution_time_sec": float(t_evo),
        "final_full_evaluation_evals": int(final_full_evals),
        "final_full_time_sec": float(t_final),
        "total_time_sec": float(total_elapsed),
        "total_evaluations_with_final_and_char": int(total_with_final),
        "saving_vs_full_percent": float(reduction_percent),
        "time_breakdown": {
            "data_partition": float(data_time),
            "characterization": float(t_char),
            "similarity_matching": float(t_sim_total),
            "evolution": float(t_evo),
            "final_full": float(t_final),
            "total": float(total_elapsed),
        },
    }
    with open(output_dir / "cost.json", "w") as f:
        json.dump(cost, f, indent=2)
    with open(output_dir / "final_evaluation" / "cost_breakdown.json", "w") as f:
        json.dump(cost, f, indent=2)

    # Save per-generation cost accumulation for plots
    if not df_gen.empty:
        df_gen["cumulative_evals"] = df_gen["eval_count"]
        df_gen["evals_avoided_cumulative"] = theoretical_full_evals * (df_gen["generation"] / cfg.evolution.n_generations) - df_gen["eval_count"]
        # theoretical cumulative at gen g: g * pop * N
        df_gen["theoretical_full_cumulative"] = df_gen["generation"] * cfg.evolution.pop_size * cfg.dataset.n_clients
        df_gen["reduction_percent"] = (1 - df_gen["eval_count"] / df_gen["theoretical_full_cumulative"]) * 100
        df_gen.to_csv(output_dir / "generations" / "cost_per_generation.csv", index=False)

    # Summary for this run
    summary = {
        "seed": int(cfg.seed),
        "best_fitness_evolution": float(best_fitness),
        "best_generation": int(best_gen),
        "final_mean_accuracy": float(df_best.iloc[0]["mean_accuracy"]) if not df_best.empty else 0.0,
        "final_worst_accuracy": float(df_best.iloc[0]["worst_client_accuracy"]) if not df_best.empty else 0.0,
        "final_std_accuracy": float(df_best.iloc[0]["std_accuracy"]) if not df_best.empty else 0.0,
        "cost": cost,
        "pairs": pairs_cids,
        "singletons": singletons_cids,
        "similarity_stats": sim_stats,
        "n_reps": int(n_reps),
        "pairing_mode": pairing_mode,
        "pairing_tau": pairing_tau,
        "rep_mode": rep_mode,
        "full_mode": full_mode,
        "prob_info": prob_info,
    }
    # enrich with final per-client accuracies
    if not df_best.empty:
        summary["final_per_client_accuracy"] = {f"client_{cid}": float(df_best.iloc[0][f"acc_client_{cid}"]) for cid in sorted(clients.keys()) if f"acc_client_{cid}" in df_best.columns}

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Metadata
    metadata = {
        "start": t_start,
        "end": time.time(),
        "elapsed_seconds": float(total_elapsed),
        "device": device,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "config": dataclasses.asdict(cfg),
        "input_dim": input_dim,
        "n_classes": n_classes,
        "pairs": pairs_cids,
        "singletons": singletons_cids,
        "n_reps": int(n_reps),
        "pairing_mode": pairing_mode,
        "pairing_tau": pairing_tau,
        "rep_mode": rep_mode,
        "full_mode": full_mode,
        "best_genome": best_genome.to_dict() if 'best_genome' in locals() and best_genome is not None else None,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Try to generate plots (best effort)
    try:
        from .plots_similarity import generate_all_plots_similarity
        generate_all_plots_similarity(output_dir, cfg)
    except Exception as e:
        print(f"[WARN] Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n[Similarity] Run finished seed {cfg.seed}")
    print(f"  Output: {output_dir}")
    print(f"  Best evolution fitness (mean rep accuracy): {best_fitness:.4f} gen {best_gen}")
    if not df_best.empty:
        print(f"  Final full mean accuracy: {df_best.iloc[0]['mean_accuracy']:.4f} worst {df_best.iloc[0]['worst_client_accuracy']:.4f}")
    print(f"  Cost saving vs full: {reduction_percent:.1f}% ({total_search_evals} vs {theoretical_full_evals} evals)")
    print(f"  Total time: {total_elapsed:.1f}s")

    return {
        "output_dir": output_dir,
        "evaluator": evaluator,
        "similarity_matrix": S,
        "pairs": pairs,
        "singletons": singletons,
        "best_genome": best_genome if 'best_genome' in locals() else None,
        "summary": summary,
        "df_generation": df_gen,
        "df_individual": df_ind,
        "df_final": df_final,
        "cost": cost,
    }
