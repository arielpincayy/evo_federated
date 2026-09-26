"""Runner federado unificado: NAS exterior + FedAvg interior.

Cada arquitectura -> modelo global FedAvg -> fitness en validation (media/mínimo).
Test oficial separado, reservado a evaluación final.

Estrategias de participación (re-prueban A/B/C + controles):
  full: todos los clientes cada ronda
  random-k: k aleatorios por ronda
  similarity: pares max-similitud, representantes alternantes por ronda (idea B)
  dissimilarity: pares max-disimilitud, representantes alternantes por ronda (idea A)
  prob-similarity / prob-dissimilarity: pareo probabilístico con tau (verificable)

Optimizadores:
  nsga2: 2 objetivos (-media_val, -min_val) vía FederatedEvaluator
  random: muestreo aleatorio con mismo presupuesto
  fixed: una MLP fija de referencia
  nsga3/moead: N objetivos por cliente sobre el mismo modelo global (idea C)
"""
import dataclasses
import hashlib
import ast
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..utils.seed import set_seed
from ..utils.config import ExperimentConfig, save_config
from ..data.federated_split import build_federated_clients
from ..federated.characterization import characterize_clients, compute_delta_vectors
from ..federated.similarity import (
    compute_similarity_matrix,
    compute_distance_matrix_from_similarity,
)
from ..federated.matching import (
    max_similarity_matching,
    max_dissimilarity_matching,
    probabilistic_similarity_pairing,
    probabilistic_dissimilarity_pairing,
)
from ..evolution.federated_evaluator import FederatedEvaluator
from ..federated.fedavg import run_fedavg, evaluate_global_on_split
from ..models.genome import Genome, genome_to_model
from ..federated.characterization import flatten_params, set_model_params
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _limit_participants(representatives, k, client_ids):
    """Return all pair representatives without silently dropping a pair."""
    target = min(max(1, int(k)), len(client_ids))
    selected = list(dict.fromkeys(int(cid) for cid in representatives))
    if len(selected) != target:
        raise ValueError(
            "Pair-based participation requires one representative per pair: "
            f"got {len(selected)} representatives for k={target}"
        )
    return sorted(selected)


def _write_dataset_artifacts(output_dir: Path, dinfo):
    train_part = dinfo["train_partition"]
    test_part = dinfo["test_partition"]
    payload = {
        "input_dim": int(dinfo["input_dim"]),
        "n_classes": int(dinfo["n_classes"]),
        "train_seed": int(train_part.seed),
        "test_seed": int(test_part.seed),
        "train_partition": train_part.to_dict(),
        "test_partition": test_part.to_dict(),
    }
    with open(output_dir / "dataset" / "partition.json", "w") as f:
        json.dump(payload, f, indent=2, default=int)
    for name, part in (("train", train_part), ("test", test_part)):
        class_dist = np.asarray(part.class_distribution)
        columns = [f"class_{i}" for i in range(class_dist.shape[1])]
        pd.DataFrame(class_dist, columns=columns).to_csv(
            output_dir / "dataset" / f"{name}_class_distribution.csv", index_label="client"
        )
        rows = []
        for cid, splits in sorted(part.splits.items()):
            rows.append({"client": int(cid), **{f"n_{k}": len(v) for k, v in splits.items()}})
        pd.DataFrame(rows).to_csv(output_dir / "dataset" / f"{name}_sizes.csv", index=False)


def _write_characterization_artifacts(output_dir: Path, char_res):
    np.save(output_dir / "characterization" / "delta_vectors.npy", compute_delta_vectors(char_res["deltas"]))
    np.save(output_dir / "characterization" / "theta0.npy", char_res["theta0"])
    pd.DataFrame.from_dict(char_res["metrics"], orient="index").to_csv(
        output_dir / "characterization" / "probe_metrics.csv", index_label="client"
    )
    with open(output_dir / "characterization" / "probe.json", "w") as f:
        json.dump({
            "genome": char_res["probe_genome"].to_dict(),
            "n_params": int(char_res["n_params"]),
            "client_ids": sorted(int(cid) for cid in char_res["deltas"]),
        }, f, indent=2)


def _validation_pareto(df):
    if df.empty:
        return df.copy()
    unique = df.drop_duplicates(subset=["genome"]).reset_index(drop=True)
    objectives = -unique[["mean_accuracy", "min_accuracy"]].to_numpy(dtype=float)
    indices = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    return unique.iloc[indices].copy().sort_values(
        ["mean_accuracy", "min_accuracy"], ascending=False
    )


def _write_schedule_artifacts(output_dir: Path, history):
    rows = []
    for record in history:
        participants = ast.literal_eval(record["participants"])
        for round_idx, cids in enumerate(participants):
            for cid in cids:
                rows.append({
                    "generation": int(record["generation"]),
                    "individual_idx": int(record["individual_idx"]),
                    "round": int(round_idx),
                    "client": int(cid),
                })
    pd.DataFrame(rows).to_csv(output_dir / "participation_schedule.csv", index=False)


def build_participant_fn(strategy, client_ids, S=None, D=None, k=2, tau=0.5, seed=42):
    """Devuelve fn(generation, rounds, k, rng) -> schedule. Registra tau usado."""
    cids = list(client_ids)
    if strategy == "full":
        def fn(gen, rounds, kk, rng):
            return [list(cids) for _ in range(rounds)]
        fn.info = {"strategy": "full", "tau": None}
        return fn
    if strategy == "random-k":
        def fn(gen, rounds, kk, rng):
            out = []
            for _ in range(rounds):
                sel = list(rng.choice(cids, size=min(kk, len(cids)), replace=False))
                out.append(sorted(map(int, sel)))
            return out
        fn.info = {"strategy": "random-k", "k": k, "tau": None}
        return fn
    if strategy in ("similarity", "dissimilarity"):
        if S is None:
            raise ValueError(f"{strategy} necesita matriz S")
        if strategy == "similarity":
            pairs, singletons = max_similarity_matching(S)
        else:
            pairs, singletons = max_dissimilarity_matching(S, D)
        pos2cid = {pos: cid for pos, cid in enumerate(cids)}

        def fn(gen, rounds, kk, rng):
            out = []
            for r in range(rounds):
                reps = []
                for a, b in pairs:
                    rep_pos = a if ((gen + r) % 2 == 0) else b
                    reps.append(int(pos2cid[rep_pos]))
                for s in singletons:
                    reps.append(int(pos2cid[s]))
                out.append(_limit_participants(reps, kk, cids))
            return out
        fn.info = {"strategy": strategy, "pairs": pairs, "singletons": singletons, "tau": None}
        return fn
    if strategy in ("prob-similarity", "prob-dissimilarity"):
        if S is None:
            raise ValueError(f"{strategy} necesita matriz S")
        rng0 = np.random.RandomState(seed)
        if strategy == "prob-similarity":
            pairs, singletons, info = probabilistic_similarity_pairing(S, float(tau), rng0)
        else:
            pairs, singletons, info = probabilistic_dissimilarity_pairing(S, float(tau), rng0, D)
        pos2cid = {pos: cid for pos, cid in enumerate(cids)}

        def fn(gen, rounds, kk, rng):
            out = []
            for r in range(rounds):
                reps = []
                for a, b in pairs:
                    rep_pos = a if ((gen + r) % 2 == 0) else b
                    reps.append(int(pos2cid[rep_pos]))
                for s in singletons:
                    reps.append(int(pos2cid[s]))
                out.append(_limit_participants(reps, kk, cids))
            return out
        fn.info = {"strategy": strategy, "pairs": pairs, "singletons": singletons,
                   "tau": float(tau), "pairing_info": info}
        return fn
    raise ValueError(f"Estrategia desconocida: {strategy}")


class FederatedManyEvaluator(FederatedEvaluator):
    """N objetivos: 1-acc por cliente del modelo global (idea C con FedAvg)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_obj = len(self.client_ids_sorted)

    def evaluate_batch(self, X: np.ndarray):
        F = np.zeros((X.shape[0], self.n_obj))
        for i in range(X.shape[0]):
            # reutiliza evaluate_single para entrenar/evaluar y luego expande a N objs
            _, info = self.evaluate_single(X[i], individual_idx=i, arch_id=f"g{self.generation}_i{i}")
            accs = [info["per_client"][c]["accuracy"] for c in self.client_ids_sorted]
            F[i] = np.array([1.0 - a for a in accs])
        return F


def run_federated_nas(cfg: ExperimentConfig, strategy: str = "full", optimizer: str = "nsga2",
                      tau: float | None = None, output_dir: Path | None = None,
                      verbose: bool = False):
    t_start = time.time()
    set_seed(cfg.seed)
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(cfg.output_dir) / f"federated_{strategy}_{optimizer}_{ts}"
    output_dir = Path(output_dir)
    for sub in ["dataset", "characterization", "similarity", "generations", "final_evaluation"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.json")
    device = _device(cfg.train.device)

    clients, dinfo = build_federated_clients(cfg)
    cids = sorted(clients.keys())
    input_dim, n_classes = dinfo["input_dim"], dinfo["n_classes"]
    _write_dataset_artifacts(output_dir, dinfo)

    # caracterización probe para S/D (ideas A/B)
    t_char_start = time.time()
    char_res = characterize_clients(
        clients=clients, decoder_input_dim=input_dim, decoder_n_classes=n_classes,
        genome_config=dataclasses.asdict(cfg.genome), train_config=dataclasses.asdict(cfg.train),
        k_epochs=cfg.characterization.k_epochs, seed=cfg.characterization.seed, device=device,
    )
    characterization_elapsed = float(time.time() - t_char_start)
    _write_characterization_artifacts(output_dir, char_res)
    S = compute_similarity_matrix(char_res["deltas"])
    D = compute_distance_matrix_from_similarity(S)
    np.save(output_dir / "similarity" / "similarity_matrix.npy", S)
    pd.DataFrame(S).to_csv(output_dir / "similarity" / "similarity_matrix.csv", index=False)

    fed_cfg = dataclasses.asdict(cfg.federated)
    k = int(fed_cfg.get("clients_per_round", 2))
    tau_eff = float(tau) if tau is not None else float(cfg.characterization.tau)
    part_fn = build_participant_fn(strategy, cids, S=S, D=D, k=k, tau=tau_eff, seed=cfg.seed)
    participation_meta = {
        "strategy": strategy,
        "optimizer": optimizer,
        "tau": getattr(part_fn, "info", {}).get("tau"),
        "info": getattr(part_fn, "info", {}),
        "federated": fed_cfg,
    }

    n_var = 1 + cfg.genome.l_max + 3
    if optimizer in ("nsga3", "moead"):
        evaluator = FederatedManyEvaluator(
            clients, input_dim, n_classes, dataclasses.asdict(cfg.genome),
            dataclasses.asdict(cfg.train), fed_cfg, participant_fn=part_fn,
            search_split=cfg.federated.search_split, device=device, seed=cfg.evolution.seed,
        )
    else:
        evaluator = FederatedEvaluator(
            clients, input_dim, n_classes, dataclasses.asdict(cfg.genome),
            dataclasses.asdict(cfg.train), fed_cfg, participant_fn=part_fn,
            search_split=cfg.federated.search_split, device=device, seed=cfg.evolution.seed,
        )

    if optimizer == "nsga2":
        from ..evolution.nsga2 import run_nsga2
        evo = run_nsga2(evaluator, n_var, pop_size=cfg.evolution.pop_size,
                        n_generations=cfg.evolution.n_generations, seed=cfg.evolution.seed,
                        hv_ref=tuple(cfg.evolution.hv_ref_min), verbose=verbose)
        hist = evo["hv_history"]
    elif optimizer == "random":
        rng = np.random.RandomState(cfg.evolution.seed)
        for g in range(1, cfg.evolution.n_generations + 1):
            evaluator.set_generation(g)
            X = rng.rand(cfg.evolution.pop_size, n_var)
            evaluator.evaluate_batch(X)
        hist = []
    elif optimizer == "fixed":
        choices = tuple(cfg.genome.n_neurons_choices)
        fixed_genome = Genome(
            L=1,
            hidden_sizes=[int(choices[0])] * cfg.genome.l_max,
            activation=cfg.genome.activations[0],
            dropout=0.0,
            use_bn=False,
            l_max=cfg.genome.l_max,
        )
        evaluator.set_generation(1)
        evaluator.evaluate_single(
            fixed_genome.to_vector(dataclasses.asdict(cfg.genome)),
            individual_idx=0,
            arch_id="fixed_mlp",
        )
        hist = []
    elif optimizer in ("nsga3", "moead"):
        from ..evolution.nsga3_runner import run_nsga3
        from ..evolution.moead_runner import run_moead
        if optimizer == "nsga3":
            evo = run_nsga3(evaluator, n_var, pop_size=cfg.evolution.pop_size,
                            n_generations=cfg.evolution.n_generations, seed=cfg.evolution.seed)
        else:
            evo = run_moead(evaluator, n_var, pop_size=cfg.evolution.pop_size,
                            n_generations=cfg.evolution.n_generations, seed=cfg.evolution.seed)
        hist = evo["hv_history"]
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer}")

    pd.DataFrame(evaluator.generation_history).to_csv(output_dir / "generations" / "population_history.csv", index=False)
    pd.DataFrame(hist).to_csv(output_dir / "generations" / "hypervolume.csv", index=False)
    _write_schedule_artifacts(output_dir, evaluator.generation_history)

    # Selección en validation: deduplicada, mejor media, luego mejor mínimo.
    df = pd.DataFrame(evaluator.generation_history)
    if not df.empty:
        df = df.drop_duplicates(subset=["genome"]).sort_values(
            ["mean_accuracy", "min_accuracy"], ascending=False
        )
        validation_pareto = _validation_pareto(df)
        validation_pareto.to_csv(output_dir / "generations" / "validation_pareto.csv", index=False)
        top = df.head(3).copy()
        top.insert(0, "selection_rank", np.arange(1, len(top) + 1))
        top["primary_candidate"] = top["selection_rank"] == 1
        top.to_csv(output_dir / "generations" / "selected_candidates.csv", index=False)
    else:
        top = df
        validation_pareto = df

    final_mode = str(fed_cfg.get("final_participation", "full"))
    if final_mode not in ("full", "same"):
        raise ValueError("federated.final_participation must be 'full' or 'same'")
    if final_mode == "full":
        final_sched = [list(cids) for _ in range(int(fed_cfg.get("rounds", 2)))]
    else:
        final_sched = part_fn(
            0,
            int(fed_cfg.get("rounds", 2)),
            k,
            np.random.RandomState(cfg.seed + 100_000),
        )
    participation_meta["final_schedule"] = final_sched
    participation_meta["final_participation"] = final_mode

    # Evaluación final: por defecto re-entrena con full; "same" usa el régimen de búsqueda.
    final_rows = []
    final_eval_elapsed = 0.0
    final_local_trainings = 0
    final_rounds = 0
    final_communication_bytes = 0
    final_communication_messages = 0
    final_state_bytes = []
    for _, row in top.iterrows():
        gdict = json.loads(row["genome"])
        genome = Genome(L=gdict["L"], hidden_sizes=tuple(gdict["hidden_sizes"]),
                        activation=gdict["activation"], dropout=gdict["dropout"],
                        use_bn=gdict["use_bn"], l_max=cfg.genome.l_max)
        torch.manual_seed(int(hashlib.md5(json.dumps(gdict, sort_keys=True).encode()).hexdigest()[:8], 16) % (2**31 - 1))
        template = genome_to_model(genome, input_dim, n_classes)
        flat_init = flatten_params(template)
        eval_start = time.time()
        fed = run_fedavg(genome, genome_to_model, clients, input_dim, n_classes, flat_init, final_sched,
                         local_epochs=int(fed_cfg.get("local_epochs", 1)),
                         lr=float(cfg.train.lr), optimizer_name=cfg.train.optimizer, device=device)
        per_test = evaluate_global_on_split(fed["model"], clients, split=cfg.federated.final_split, device=device)
        final_eval_elapsed += float(time.time() - eval_start)
        final_local_trainings += int(fed["n_local_trainings"])
        final_rounds += int(fed["n_rounds"])
        final_communication_bytes += int(fed["communication_bytes"])
        final_communication_messages += int(fed["communication_messages"])
        final_state_bytes.append(int(fed["state_bytes"]))
        accs = [per_test[c]["accuracy"] for c in cids]
        f1s = [per_test[c]["macro_f1"] for c in cids]
        final_rows.append({
            "selection_rank": int(row.get("selection_rank", len(final_rows) + 1)),
            "primary_candidate": bool(row.get("primary_candidate", len(final_rows) == 0)),
            "genome": json.dumps(gdict), "val_mean": float(row["mean_accuracy"]),
            "val_min": float(row["min_accuracy"]),
            "test_mean_accuracy": float(np.mean(accs)), "test_min_accuracy": float(np.min(accs)),
            "test_mean_f1": float(np.mean(f1s)), "test_min_f1": float(np.min(f1s)),
            "n_local_trainings_final": int(fed["n_local_trainings"]),
            "n_rounds_final": int(fed["n_rounds"]),
            "final_participation": final_mode,
        })
        for cid, res in per_test.items():
            final_rows[-1][f"test_acc_client_{cid}"] = float(res["accuracy"])
            final_rows[-1][f"test_f1_client_{cid}"] = float(res["macro_f1"])
    if final_rows:
        final_df = pd.DataFrame(final_rows)
        final_df.to_csv(output_dir / "final_evaluation" / "final_test.csv", index=False)
        _validation_pareto(
            final_df.rename(columns={"test_mean_accuracy": "mean_accuracy", "test_min_accuracy": "min_accuracy"})
        ).to_csv(output_dir / "final_evaluation" / "final_pareto.csv", index=False)

    with open(output_dir / "participation.json", "w") as f:
        json.dump(participation_meta, f, indent=2, default=str)

    cost = {
        "strategy": strategy, "optimizer": optimizer, "tau": getattr(part_fn, "info", {}).get("tau"),
        "n_arch_evals": int(evaluator.eval_count),
        "n_local_trainings_search": int(evaluator.n_local_trainings),
        "rounds_total": int(evaluator.n_rounds_total),
        "n_local_trainings_probe": int(len(clients)),
        "probe_steps": int(sum(m.get("steps", 0) for m in char_res["metrics"].values())),
        "probe_elapsed": characterization_elapsed,
        "n_local_trainings_final": int(final_local_trainings),
        "rounds_final": int(final_rounds),
        "final_eval_elapsed": float(final_eval_elapsed),
        "n_local_trainings_total": int(evaluator.n_local_trainings + len(clients) + final_local_trainings),
        "communication_bytes_search": int(evaluator.communication_bytes_total),
        "communication_messages_search": int(evaluator.communication_messages_total),
        "communication_bytes_final": int(final_communication_bytes),
        "communication_messages_final": int(final_communication_messages),
        "communication_state_bytes_final": final_state_bytes,
        "n_final_candidates": int(len(final_rows)),
        "final_participation": final_mode,
        "search_split": cfg.federated.search_split, "final_split": cfg.federated.final_split,
        "elapsed": float(time.time() - t_start),
    }
    with open(output_dir / "cost.json", "w") as f:
        json.dump(cost, f, indent=2)
    return {"output_dir": str(output_dir), "cost": cost, "n_final": len(final_rows)}
