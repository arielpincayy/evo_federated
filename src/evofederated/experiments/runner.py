"""Experiment Runner: dataset, partition, characterization, evolution, exhaustive validation, checkpoints, Hall of Fame."""
import json
import time
from pathlib import Path
from datetime import datetime
import dataclasses
import platform
import numpy as np
import torch
import pandas as pd

from ..utils.seed import set_seed
from ..utils.config import ExperimentConfig
from ..data.datasets import get_dataset, make_synthetic_dataset, dirichlet_partition, heterogeneity_metrics, get_input_dim_and_n_classes
from ..clients.client import FederatedClient
from ..clients.server import FederatedServer
from ..federated.characterization import characterize_clients, flatten_params, set_model_params
from ..federated.divergence import compute_divergence_matrix, divergence_stats, most_divergent_pairs
from ..federated.pairing import get_pairing
from ..evolution.evaluator import Evaluator
from ..evolution.nsga2 import run_nsga2
from ..models.genome import Genome, genome_to_model
from ..evolution.validation import exhaustive_evaluate_pareto, exhaustive_evaluate_genome, compute_global_pareto, compute_hv_common, HallOfFame, genome_hash


def _get_device(train_device: str):
    if train_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return train_device


def _checkpoint_generations(n_generations: int, control_every: int):
    """Return sorted checkpoint gens including 1 and final. If control_every==0 -> only final."""
    # ponytail: simple list, no over-engineering
    if control_every is None or control_every <= 0:
        return [n_generations]
    # spec example for 20 gens: 1,5,10,15,20 when control_every=5
    base = list(range(control_every, n_generations + 1, control_every))
    gens = sorted(set([1] + base + [n_generations]))
    return [g for g in gens if 1 <= g <= n_generations]


def _extract_pareto_genomes_for_generation(pop_history, cfg_genome_dict, generation: int):
    """From pop_history (list), extract Pareto genomes for that generation."""
    recs = [r for r in pop_history if r.get("generation") == generation]
    if not recs:
        return [], np.array([])
    # F: for 2-client strategies F is stored as "F" list; for FULL it's also F. Need to recover F per individual
    Fs = []
    genomes = []
    for r in recs:
        f = r.get("F")
        if f is None:
            # try F1_i / F1_j or F1_mean/min fields
            if "F1_i" in r and "F1_j" in r:
                f = [-r["F1_i"], -r["F1_j"]]
            elif "F1_mean" in r:
                f = [-r["F1_mean"], -r["F1_min"]]
            else:
                continue
        Fs.append(f)
        # reconstruct genome from stored dict fields: genome dict may be in r["genome"] as dict
        gdict = r.get("genome")
        if isinstance(gdict, dict):
            genomes.append(Genome.from_dict(gdict))
        else:
            # fallback: genome_* fields? For simplicity create via stored F? But pop_history should have dict
            # Try to reconstruct via vector not available; skip
            genomes.append(None)
    if not Fs:
        return [], np.array([])
    Fs_arr = np.array(Fs, dtype=float)
    # Filter valid genomes (non-None)
    valid_idx = [i for i, g in enumerate(genomes) if g is not None]
    if not valid_idx:
        return [], Fs_arr
    Fs_valid = Fs_arr[valid_idx]
    genomes_valid = [genomes[i] for i in valid_idx]
    # Non-dominated sorting
    try:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds = NonDominatedSorting().do(Fs_valid, only_non_dominated_front=True)
        pareto_genomes = [genomes_valid[i] for i in nds]
        pareto_F = Fs_valid[nds]
        return pareto_genomes, pareto_F
    except Exception:
        return genomes_valid, Fs_valid


def run_experiment(cfg: ExperimentConfig, output_dir: Path = None, verbose: bool = False):
    t0 = time.time()
    set_seed(cfg.seed)

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(cfg.output_dir) / f"experiment_{ts}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (output_dir / "characterization").mkdir(parents=True, exist_ok=True)
    (output_dir / "generations").mkdir(parents=True, exist_ok=True)
    (output_dir / "architectures").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    from ..utils.config import save_config
    save_config(cfg, output_dir / "config.json")
    try:
        save_config(cfg, output_dir / "config.yaml")
    except Exception:
        pass

    device = _get_device(cfg.train.device)

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

    server = FederatedServer(clients)
    server.collect_client_info()

    het_metrics = heterogeneity_metrics(partition.class_distribution)
    with open(output_dir / "dataset" / "heterogeneity.json", "w") as f:
        json.dump({"partition": partition.to_dict(), "heterogeneity": het_metrics, "input_dim": input_dim, "n_classes": n_classes}, f, indent=2, default=str)
    pd.DataFrame(partition.class_distribution, columns=[f"class_{i}" for i in range(partition.n_classes)]).to_csv(output_dir / "dataset" / "class_distribution.csv", index_label="client")

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
    D = compute_divergence_matrix(char_res["deltas"], metric=cfg.characterization.distance_metric)
    server.set_divergence_matrix(D)
    div_stats = divergence_stats(D)
    top_pairs = most_divergent_pairs(D, top_k=10)

    np.save(output_dir / "characterization" / "divergence_matrix.npy", D)
    pd.DataFrame(D).to_csv(output_dir / "characterization" / "divergence_matrix.csv", index=False)
    with open(output_dir / "characterization" / "divergence_stats.json", "w") as f:
        json.dump({"stats": div_stats, "top_pairs": top_pairs, "probe_genome": char_res["probe_genome"].to_dict(), "n_params": char_res["n_params"]}, f, indent=2, default=str)
    delta_norms = {str(cid): float(np.linalg.norm(v)) for cid, v in char_res["deltas"].items()}
    with open(output_dir / "characterization" / "delta_norms.json", "w") as f:
        json.dump(delta_norms, f, indent=2)

    char_cost = cfg.dataset.n_clients
    props = partition.class_distribution / partition.class_distribution.sum(axis=1, keepdims=True).clip(min=1)
    n = cfg.dataset.n_clients
    true_D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            true_D[i, j] = np.abs(props[i] - props[j]).sum() / 2
    try:
        from scipy.stats import spearmanr, kendalltau
        idx = np.triu_indices(n, k=1)
        d_flat = D[idx]
        true_flat = true_D[idx]
        spearman_corr, _ = spearmanr(d_flat, true_flat)
        kendall_corr, _ = kendalltau(d_flat, true_flat)
    except Exception:
        spearman_corr = 0.0
        kendall_corr = 0.0
    try:
        pearson_corr = float(np.corrcoef(d_flat, true_flat)[0, 1])
    except Exception:
        pearson_corr = 0.0
    with open(output_dir / "characterization" / "proxy_validation.json", "w") as f:
        json.dump({"spearman": float(spearman_corr) if spearman_corr is not None else None,
                   "kendall": float(kendall_corr) if kendall_corr is not None else None,
                   "pearson": float(pearson_corr),
                   "note": "correlation between Delta-based D and true TV. Offline only."}, f, indent=2)
    np.save(output_dir / "characterization" / "true_TV_matrix.npy", true_D)

    l_max = cfg.genome.l_max
    n_var = 1 + l_max + 3
    results_per_strategy = {}
    hv_comparison = []
    exhaustive_validations = {}

    # Global Hall of Fame across strategies? We'll keep per experiment combined and per strategy
    experiment_hof = HallOfFame()

    for strategy in cfg.baselines:
        print(f"\n=== Running strategy: {strategy} ===")
        pairing = get_pairing(strategy, tau=cfg.evolution.tau if strategy == "dynamic" else cfg.characterization.tau)
        evaluator = Evaluator(
            clients=clients,
            input_dim=input_dim,
            n_classes=n_classes,
            genome_config=dataclasses.asdict(cfg.genome),
            train_config=dataclasses.asdict(cfg.train),
            pairing_strategy=pairing,
            divergence_matrix=D,
            strategy_name=strategy,
            seed=cfg.evolution.seed,
            device=device,
            generation=0,
        )
        hv_ref = cfg.evolution.hv_ref_min
        # snapshot rng state for isolation check later
        import copy as _copy
        # store evolution config seed and evaluator initial state
        _eval_count_before = evaluator.eval_count

        run_res = run_nsga2(
            evaluator=evaluator,
            n_var=n_var,
            pop_size=cfg.evolution.pop_size,
            n_generations=cfg.evolution.n_generations,
            crossover_prob=cfg.evolution.crossover_prob,
            crossover_eta=cfg.evolution.crossover_eta,
            mutation_prob=cfg.evolution.mutation_prob,
            mutation_eta=cfg.evolution.mutation_eta,
            seed=cfg.evolution.seed,
            hv_ref=hv_ref,
            verbose=verbose,
        )
        strat_dir = output_dir / f"strategy_{strategy}"
        strat_dir.mkdir(parents=True, exist_ok=True)
        (strat_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        hv_hist = run_res["hv_history"]
        df_hv = pd.DataFrame(hv_hist)
        df_hv["strategy"] = strategy
        df_hv["seed"] = cfg.seed
        df_hv["hv_type"] = "internal"
        df_hv["ref_point"] = str(hv_ref)
        df_hv.to_csv(strat_dir / "hypervolume_history.csv", index=False)
        df_hv.to_csv(output_dir / "generations" / f"hypervolume_{strategy}.csv", index=False)

        pair_hist = evaluator.pair_history
        pd.DataFrame(pair_hist).to_csv(strat_dir / "pair_history.csv", index=False)
        pd.DataFrame(pair_hist).to_csv(output_dir / "generations" / f"pair_history_{strategy}.csv", index=False)

        pop_hist = evaluator.pop_history
        pop_records = []
        for rec in pop_hist:
            r = {k: v for k, v in rec.items() if k not in ("genome",)}
            g = rec.get("genome", {})
            if isinstance(g, dict):
                for gk, gv in g.items():
                    r[f"genome_{gk}"] = str(gv) if isinstance(gv, list) else gv
            pop_records.append(r)
        pd.DataFrame(pop_records).to_csv(strat_dir / "population_history.csv", index=False)

        res = run_res["result"]
        pop = res.pop
        F = pop.get("F") if pop is not None else None
        X = pop.get("X") if pop is not None else None
        try:
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            nds_idx = NonDominatedSorting().do(F, only_non_dominated_front=True) if F is not None else []
            F_pf = F[nds_idx] if len(nds_idx) else np.array([])
            X_pf = X[nds_idx] if len(nds_idx) else np.array([])
        except Exception:
            F_pf = F
            X_pf = X
            nds_idx = []

        pf_rows = []
        final_genomes = []
        for i, (f_vec, x_vec) in enumerate(zip(F_pf, X_pf)):
            genome = Genome.from_vector(x_vec, dataclasses.asdict(cfg.genome))
            final_genomes.append(genome)
            pf_rows.append({
                "idx": int(i),
                "F0": float(f_vec[0]),
                "F1": float(f_vec[1]),
                "F1_i_est": -float(f_vec[0]),
                "F1_j_est": -float(f_vec[1]),
                "genome": str(genome.to_dict()),
                "genome_hash": genome_hash(genome),
                "genome_L": genome.L,
                "genome_hidden": str(genome.hidden_sizes),
                "genome_act": genome.activation,
                "genome_dropout": genome.dropout,
                "genome_bn": genome.use_bn,
            })
        pd.DataFrame(pf_rows).to_csv(strat_dir / "final_pareto.csv", index=False)
        pd.DataFrame(pf_rows).to_csv(output_dir / "architectures" / f"pareto_{strategy}.csv", index=False)

        # Exhaustive validation of final Pareto (isolated, does not affect evaluator)
        search_evals = int(evaluator.eval_count)
        # checkpoint evaluations will be tracked separately
        checkpoint_evals = 0
        final_validation_evals = 0

        # Use validation module for final
        df_exh = exhaustive_evaluate_pareto(final_genomes, clients, input_dim, n_classes, dataclasses.asdict(cfg.train), device=device, strategy=strategy, generation=cfg.evolution.n_generations)
        if not df_exh.empty:
            df_exh["strategy"] = strategy
            final_validation_evals = len(df_exh) * len(clients)
        else:
            df_exh = pd.DataFrame()
        df_exh.to_csv(strat_dir / "exhaustive_validation.csv", index=False)
        exhaustive_validations[strategy] = df_exh

        # Additional per-spec files: pareto_all_clients_f1.csv and accuracy
        if not df_exh.empty:
            f1_cols = [c for c in df_exh.columns if c.startswith("f1_client_")]
            acc_cols = [c for c in df_exh.columns if c.startswith("acc_client_")]
            # F1 matrix with genome info
            df_f1 = df_exh[["pareto_idx", "genome_hash", "mean_f1", "median_f1", "min_f1", "max_f1", "std_f1"] + f1_cols]
            df_f1.to_csv(strat_dir / "pareto_all_clients_f1.csv", index=False)
            df_f1.to_csv(output_dir / "architectures" / f"pareto_all_clients_f1_{strategy}.csv", index=False)
            df_acc = df_exh[["pareto_idx", "genome_hash", "mean_acc", "min_acc"] + acc_cols]
            df_acc.to_csv(strat_dir / "pareto_all_clients_accuracy.csv", index=False)
            # also combined at top level? keep per strategy
        # Compute HV common validation
        if not df_exh.empty:
            ref_common = np.array([0.1, 0.1]) if np.allclose(np.array(hv_ref), 0) else np.array(hv_ref)
            hv_common = compute_hv_common(df_exh, ref_common)
        else:
            hv_common = 0.0
            ref_common = np.array(hv_ref)
        with open(strat_dir / "hv_common.json", "w") as f:
            json.dump({"hv_common_validation": float(hv_common), "ref_common": ref_common.tolist() if hasattr(ref_common, "tolist") else ref_common, "n_pareto": len(df_exh)}, f, indent=2)

        # Global Pareto distinction
        if not df_exh.empty:
            is_global, hv_global, F_global = compute_global_pareto(df_exh, ref_common)
            df_exh["is_global_pareto"] = is_global
            df_exh["global_pareto_rank"] = is_global.astype(int)
            # save global pareto subset
            df_global = df_exh[is_global].copy()
            df_global.to_csv(strat_dir / "global_pareto.csv", index=False)
            # also save pareto front in evolution vs Exhaustively Validated distinction
            # Add column to final_pareto to mark vs global? We'll keep separate
        else:
            is_global = np.array([])
            df_global = pd.DataFrame()

        # Checkpoints exhaustive validation (analysis only, isolated)
        # Determine checkpoint gens
        ckpt_gens = _checkpoint_generations(cfg.evolution.n_generations, cfg.evolution.control_every)
        # ponytail: also ensure 1 is always included even if control_every=0? For now logic gives [N] only, add 1 if >1 and not present for analysis
        # We'll adapt: if only final, also add 1 if N>1 for minimal checkpoint coverage? But spec says for 20 gens include 1,5,... So we keep as is.
        # For small experiments without control_every, we still want at least 1 checkpoint at final (already) – treat as final only.
        strategy_hof = HallOfFame()
        checkpoint_rows = []
        checkpoint_summary = []
        # For rigorous isolation, snapshot evaluator state before checkpoints
        eval_count_snapshot = evaluator.eval_count
        rng_state_snapshot = evaluator.rng.get_state() if hasattr(evaluator.rng, "get_state") else None
        pop_history_snapshot = list(evaluator.pop_history)  # copy
        # Now perform checkpoint exhaustive evaluations without touching evaluator
        for gen in ckpt_gens:
            # avoid double evaluating final if we already did final? We'll still do but reuse df_exh for final gen to avoid double cost
            if gen == cfg.evolution.n_generations and not df_exh.empty:
                df_ckpt = df_exh.copy()
                df_ckpt["checkpoint_generation"] = gen
                # but need to avoid counting final again as checkpoint? We'll count checkpoint_evals separately but exclude final count if same?
                # For final, checkpoint and final overlap; we will not double count final as checkpoint_evals if gen==final and we reuse.
                # Instead treat checkpoint final as same as final_validation, not extra.
                # To keep separation, we will not add to checkpoint_evals for final duplicate.
                df_ckpt.to_csv(strat_dir / "checkpoints" / f"checkpoint_gen{gen}.csv", index=False)
                checkpoint_rows.append(df_ckpt)
                # Compute HV common for checkpoint
                hv_ckpt = hv_common  # same
                checkpoint_summary.append({"generation": gen, "n_archs": len(df_ckpt), "hv_common": float(hv_ckpt), "mean_of_mean_f1": float(df_ckpt["mean_f1"].mean()) if len(df_ckpt) else 0, "best_mean_f1": float(df_ckpt["mean_f1"].max()) if len(df_ckpt) else 0, "best_min_f1": float(df_ckpt["min_f1"].max()) if len(df_ckpt) else 0})
                # Hall of fame update
                for _, row in df_ckpt.iterrows():
                    # reconstruct genome from hash? Need genome object – pull from final_genomes by idx
                    # For checkpoint final we have mapping via pareto_idx, retrieve genome from final_genomes
                    idx = int(row["pareto_idx"])
                    if 0 <= idx < len(final_genomes):
                        g = final_genomes[idx]
                        # fitness from pf_rows? Use F_pf
                        f = F_pf[idx] if idx < len(F_pf) else np.array([0,0])
                        metrics = {"mean_f1": row["mean_f1"], "median_f1": row["median_f1"], "min_f1": row["min_f1"], "max_f1": row["max_f1"], "std_f1": row["std_f1"], "mean_acc": row["mean_acc"]}
                        per_client = {cid: row[f"f1_client_{cid}"] for cid in sorted(clients.keys()) if f"f1_client_{cid}" in row}
                        strategy_hof.add(g, gen, strategy, cfg.seed, f, metrics, per_client)
                        experiment_hof.add(g, gen, strategy, cfg.seed, f, metrics, per_client)
                continue
            # For other gens, extract pareto genomes for that generation
            ckpt_genomes, ckpt_F = _extract_pareto_genomes_for_generation(pop_hist, dataclasses.asdict(cfg.genome), gen)
            if not ckpt_genomes:
                checkpoint_summary.append({"generation": gen, "n_archs": 0, "hv_common": 0.0, "mean_of_mean_f1": 0, "best_mean_f1": 0, "best_min_f1": 0})
                continue
            df_ckpt = exhaustive_evaluate_pareto(ckpt_genomes, clients, input_dim, n_classes, dataclasses.asdict(cfg.train), device=device, strategy=strategy, generation=gen)
            if not df_ckpt.empty:
                df_ckpt["checkpoint_generation"] = gen
                df_ckpt["strategy"] = strategy
                df_ckpt.to_csv(strat_dir / "checkpoints" / f"checkpoint_gen{gen}.csv", index=False)
                checkpoint_rows.append(df_ckpt)
                # HV
                hv_ckpt = compute_hv_common(df_ckpt, ref_common)
                checkpoint_summary.append({"generation": gen, "n_archs": len(df_ckpt), "hv_common": float(hv_ckpt), "mean_of_mean_f1": float(df_ckpt["mean_f1"].mean()), "best_mean_f1": float(df_ckpt["mean_f1"].max()), "best_min_f1": float(df_ckpt["min_f1"].max())})
                checkpoint_evals += len(df_ckpt) * len(clients)
                # Hall of Fame
                for i, (g, f_vec) in enumerate(zip(ckpt_genomes, ckpt_F)):
                    row = df_ckpt.iloc[i]
                    metrics = {"mean_f1": row["mean_f1"], "median_f1": row["median_f1"], "min_f1": row["min_f1"], "max_f1": row["max_f1"], "std_f1": row["std_f1"], "mean_acc": row["mean_acc"]}
                    per_client = {cid: row[f"f1_client_{cid}"] for cid in sorted(clients.keys()) if f"f1_client_{cid}" in row}
                    strategy_hof.add(g, gen, strategy, cfg.seed, f_vec, metrics, per_client)
                    experiment_hof.add(g, gen, strategy, cfg.seed, f_vec, metrics, per_client)
            else:
                checkpoint_summary.append({"generation": gen, "n_archs": 0, "hv_common": 0.0, "mean_of_mean_f1": 0, "best_mean_f1": 0, "best_min_f1": 0})

        # Verify isolation: evaluator state unchanged
        assert evaluator.eval_count == eval_count_snapshot, "Checkpoint altered eval_count!"
        if rng_state_snapshot is not None:
            # np RandomState get_state compares
            assert np.array_equal(evaluator.rng.get_state()[1], rng_state_snapshot[1]), "Checkpoint altered RNG!"
        assert len(evaluator.pop_history) == len(pop_history_snapshot), "Checkpoint altered pop_history!"

        # Save checkpoint summary
        if checkpoint_summary:
            pd.DataFrame(checkpoint_summary).to_csv(strat_dir / "checkpoint_summary.csv", index=False)
            pd.DataFrame(checkpoint_summary).to_csv(strat_dir / "checkpoints" / "summary.csv", index=False)
        # Save combined checkpoint exhaustive
        if checkpoint_rows:
            df_all_ckpt = pd.concat(checkpoint_rows, ignore_index=True)
            df_all_ckpt.to_csv(strat_dir / "checkpoints" / "all_checkpoints_exhaustive.csv", index=False)
        else:
            df_all_ckpt = pd.DataFrame()

        # Hall of Fame per strategy
        df_hof = strategy_hof.to_dataframe()
        df_hof.to_csv(strat_dir / "hall_of_fame.csv", index=False)
        # Also store stats for HOF
        hof_stats = {}
        if not df_hof.empty:
            # compute global pareto among HOF
            is_hof_global, hv_hof, _ = compute_global_pareto(df_hof, ref_common)
            df_hof["is_global_pareto"] = is_hof_global
            df_hof.to_csv(strat_dir / "hall_of_fame.csv", index=False)
            hof_stats = {
                "size": int(len(df_hof)),
                "global_pareto_size": int(is_hof_global.sum()) if len(is_hof_global) else 0,
                "hv_common": float(hv_hof),
                "best_mean_f1": float(df_hof["mean_f1"].max()),
                "best_min_f1": float(df_hof["min_f1"].max()),
            }
        else:
            hof_stats = {"size": 0, "global_pareto_size": 0, "hv_common": 0.0}

        with open(strat_dir / "hall_of_fame_stats.json", "w") as f:
            json.dump(hof_stats, f, indent=2)

        # Save heatmap data for Pareto All clients: already have csv, now ensure plots will generate heatmaps
        # (plots module will handle)

        results_per_strategy[strategy] = {
            "hv_history": hv_hist,
            "hv_final_internal": float(hv_hist[-1]["hv"]) if hv_hist else 0.0,
            "hv_common": float(hv_common),
            "eval_count": search_evals,
            "checkpoint_evals": int(checkpoint_evals),
            "final_validation_evals": int(final_validation_evals),
            "exhaustive_df": df_exh,
            "F_pf": F_pf,
            "hof": df_hof,
            "checkpoint_summary": checkpoint_summary,
        }
        hv_comparison.append(df_hv)

        from ..metrics.cost import cost_metrics
        # cost includes search + char, but checkpoint and final are extra analysis cost
        total_search = search_evals + char_cost
        cost = cost_metrics(total_search, cfg.evolution.n_generations, cfg.evolution.pop_size, cfg.dataset.n_clients, strategy, characterization_cost=char_cost)
        # add separated costs
        cost.update({
            "search_evaluations": int(search_evals),
            "characterization_evaluations": int(char_cost),
            "checkpoint_evaluations": int(checkpoint_evals),
            "final_validation_evaluations": int(final_validation_evals),
            "total_analysis_evaluations": int(checkpoint_evals + final_validation_evals),
            "total_evaluations": int(total_search + checkpoint_evals + final_validation_evals),
        })
        with open(strat_dir / "cost.json", "w") as f:
            json.dump(cost, f, indent=2)

    # Save combined hv comparison
    if hv_comparison:
        df_all_hv = pd.concat(hv_comparison, ignore_index=True)
        df_all_hv["method"] = df_all_hv["strategy"]
        df_all_hv["HV"] = df_all_hv["hv"]
        df_all_hv["cumulative_evals"] = df_all_hv["eval_count"]
        df_all_hv["cumulative_time"] = df_all_hv["elapsed"]
        df_all_hv.to_csv(output_dir / "hypervolume_history.csv", index=False)
        df_all_hv.to_csv(output_dir / "generations" / "hypervolume_history.csv", index=False)

    if exhaustive_validations:
        dfs = [df for df in exhaustive_validations.values() if not df.empty]
        if dfs:
            df_all_exh = pd.concat(dfs, ignore_index=True)
            # compute global pareto flag across combined? But per strategy already flagged; for combined compute overall global Pareto across all strategies for analysis
            ref_common_comb = np.array([0.1, 0.1])
            try:
                is_comb_global, hv_comb, _ = compute_global_pareto(df_all_exh, ref_common_comb)
                df_all_exh["is_combined_global_pareto"] = is_comb_global
            except Exception:
                pass
            df_all_exh.to_csv(output_dir / "exhaustive_validation.csv", index=False)
            # Combined global pareto
            df_comb_global = df_all_exh[df_all_exh["is_combined_global_pareto"]] if "is_combined_global_pareto" in df_all_exh.columns else pd.DataFrame()
            if not df_comb_global.empty:
                df_comb_global.to_csv(output_dir / "architectures" / "global_pareto_combined.csv", index=False)
            # Also pareto_all_clients_f1 combined
            f1_cols_comb = [c for c in df_all_exh.columns if c.startswith("f1_client_")]
            if f1_cols_comb:
                df_f1_comb = df_all_exh[["strategy", "pareto_idx", "genome_hash", "mean_f1", "median_f1", "min_f1", "max_f1", "std_f1"] + f1_cols_comb]
                df_f1_comb.to_csv(output_dir / "pareto_all_clients_f1.csv", index=False)
                acc_cols = [c for c in df_all_exh.columns if c.startswith("acc_client_")]
                if acc_cols:
                    df_acc_comb = df_all_exh[["strategy", "pareto_idx", "genome_hash", "mean_acc", "min_acc"] + acc_cols]
                    df_acc_comb.to_csv(output_dir / "pareto_all_clients_accuracy.csv", index=False)

    pop_frames = []
    pair_frames = []
    pareto_frames = []
    for strat in cfg.baselines:
        strat_dir = output_dir / f"strategy_{strat}"
        p_path = strat_dir / "population_history.csv"
        if p_path.exists():
            df = pd.read_csv(p_path)
            df["strategy"] = strat
            pop_frames.append(df)
        pair_path = strat_dir / "pair_history.csv"
        if pair_path.exists():
            df = pd.read_csv(pair_path)
            df["method"] = strat
            pair_frames.append(df)
        pareto_path = strat_dir / "final_pareto.csv"
        if pareto_path.exists():
            df = pd.read_csv(pareto_path)
            df["strategy"] = strat
            pareto_frames.append(df)

    if pop_frames:
        df_pop_all = pd.concat(pop_frames, ignore_index=True)
        df_pop_all.to_csv(output_dir / "population_history.csv", index=False)
        df_pop_all.to_csv(output_dir / "generations" / "population_history.csv", index=False)
        df_pop_all.to_csv(output_dir / "evaluations.csv", index=False)
    if pair_frames:
        df_pair_all = pd.concat(pair_frames, ignore_index=True)
        df_pair_all.to_csv(output_dir / "pair_history.csv", index=False)
    if pareto_frames:
        df_pareto_all = pd.concat(pareto_frames, ignore_index=True)
        df_pareto_all.to_csv(output_dir / "final_pareto.csv", index=False)
        df_pareto_all.to_csv(output_dir / "architectures" / "final_pareto.csv", index=False)

    # Save experiment Hall of Fame combined
    df_exp_hof = experiment_hof.to_dataframe()
    if not df_exp_hof.empty:
        # compute global pareto for experiment HOF
        ref_common = np.array([0.1, 0.1])
        is_hof_global, hv_hof_exp, _ = compute_global_pareto(df_exp_hof, ref_common)
        df_exp_hof["is_global_pareto"] = is_hof_global
    df_exp_hof.to_csv(output_dir / "hall_of_fame.csv", index=False)
    df_exp_hof.to_csv(output_dir / "architectures" / "hall_of_fame.csv", index=False)

    # Checkpoint aggregated at experiment level
    ckpt_all_frames = []
    for strat in cfg.baselines:
        strat_dir = output_dir / f"strategy_{strat}"
        ckpt_path = strat_dir / "checkpoints" / "all_checkpoints_exhaustive.csv"
        if ckpt_path.exists():
            try:
                df = pd.read_csv(ckpt_path)
                ckpt_all_frames.append(df)
            except Exception:
                pass
    if ckpt_all_frames:
        df_ckpt_all = pd.concat(ckpt_all_frames, ignore_index=True)
        df_ckpt_all.to_csv(output_dir / "checkpoints" / "checkpoints_all_exhaustive.csv", index=False)
        # also summarize progression for plots
        df_ckpt_summary_all = df_ckpt_all.groupby(["strategy", "checkpoint_generation"]).agg(
            n_archs=("mean_f1", "size"),
            mean_of_mean_f1=("mean_f1", "mean"),
            best_mean_f1=("mean_f1", "max"),
            best_min_f1=("min_f1", "max"),
            mean_min_f1=("min_f1", "mean"),
        ).reset_index()
        df_ckpt_summary_all.to_csv(output_dir / "checkpoints" / "checkpoints_summary.csv", index=False)
        # Compute global quality evolution: HV per checkpoint per strategy
        hv_rows = []
        for (strat, gen), sub in df_ckpt_all.groupby(["strategy", "checkpoint_generation"]):
            hv = compute_hv_common(sub, np.array([0.1, 0.1]))
            hv_rows.append({"strategy": strat, "generation": gen, "hv_common": hv, "n_archs": len(sub)})
        pd.DataFrame(hv_rows).to_csv(output_dir / "checkpoints" / "hv_vs_generation_checkpoints.csv", index=False)

    summary_rows = []
    for strat, res in results_per_strategy.items():
        df_exh = res["exhaustive_df"]
        if df_exh is not None and not df_exh.empty:
            mean_mean_f1 = float(df_exh["mean_f1"].mean())
            median_mean_f1 = float(df_exh["mean_f1"].median())
            best_mean_f1 = float(df_exh["mean_f1"].max())
            worst_min_f1 = float(df_exh["min_f1"].min())
            mean_min_f1 = float(df_exh["min_f1"].mean())
            best_min_f1 = float(df_exh["min_f1"].max())
            median_min_f1 = float(df_exh["min_f1"].median())
            std_mean = float(df_exh["mean_f1"].std())
        else:
            mean_mean_f1 = median_mean_f1 = best_mean_f1 = worst_min_f1 = mean_min_f1 = best_min_f1 = median_min_f1 = std_mean = 0.0
        summary_rows.append({
            "strategy": strat,
            "hv_final_internal": res["hv_final_internal"],
            "hv_common_validation": res["hv_common"],
            "eval_count": res["eval_count"] + char_cost,
            "eval_count_evolution_only": res["eval_count"],
            "char_cost": char_cost,
            "checkpoint_evaluations": res.get("checkpoint_evals", 0),
            "final_validation_evaluations": res.get("final_validation_evals", 0),
            "search_evaluations": res.get("eval_count", 0),
            "total_evaluations": res.get("eval_count", 0) + char_cost + res.get("checkpoint_evals", 0) + res.get("final_validation_evals", 0),
            "mean_of_mean_f1_exhaustive": mean_mean_f1,
            "median_of_mean_f1": median_mean_f1,
            "best_mean_f1": best_mean_f1,
            "mean_of_min_f1": mean_min_f1,
            "median_of_min_f1": median_min_f1,
            "worst_min_f1": worst_min_f1,
            "best_min_f1": best_min_f1,
            "std_of_mean_f1": std_mean,
            "hof_size": len(res.get("hof", pd.DataFrame())),
            "global_pareto_size": int(df_exh["is_global_pareto"].sum()) if df_exh is not None and "is_global_pareto" in df_exh.columns and not df_exh.empty else 0,
        })
    df_summary = pd.DataFrame(summary_rows)
    if "full" in results_per_strategy:
        full_evals = results_per_strategy["full"]["eval_count"] + char_cost
        df_summary["saving_vs_full"] = df_summary["eval_count"].apply(lambda x: 1 - x / full_evals if full_evals else 0)
        df_summary["saving_percent"] = df_summary["saving_vs_full"] * 100
        # saving vs total with checkpoint/final excluded? Report saving for search only (scientific)
        df_summary["saving_search_vs_full"] = df_summary["eval_count"].apply(lambda x: 1 - x / full_evals if full_evals else 0)
    df_summary.to_csv(output_dir / "summary.csv", index=False)

    end_time = time.time()
    metadata = {
        "start": t0,
        "end": end_time,
        "elapsed_seconds": end_time - t0,
        "device": device,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "config": dataclasses.asdict(cfg),
        "input_dim": input_dim,
        "n_classes": n_classes,
    }
    try:
        import pymoo
        metadata["pymoo_version"] = pymoo.__version__
    except Exception:
        pass
    try:
        import numpy
        metadata["numpy_version"] = numpy.__version__
    except Exception:
        pass
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    try:
        from ..experiments.plots import generate_all_plots
        generate_all_plots(output_dir, cfg)
    except Exception as e:
        print(f"[WARN] Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    return {
        "output_dir": output_dir,
        "partition": partition,
        "D": D,
        "results_per_strategy": results_per_strategy,
        "summary": df_summary,
        "metadata": metadata,
    }
