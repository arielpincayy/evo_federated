"""Una corrida completa: datos -> probe/clusters -> búsqueda -> reevaluación final del frente en test."""
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.federated_split import build_federated_clients
from ..evolution.cs_evaluator import ClusterSampleEvaluator
from ..evolution.cs_nas import HV_REF, _hv, output_front, run_search
from ..federated.characterization import probe_deltas
from ..federated.clustering import cluster_delta, cluster_labels_oracle, cluster_quality, cluster_random
from ..models.cnn_genome import CNNGenome, CNNSpace, cnn_to_model, max_n_params
from ..utils.config import ExperimentConfig, save_config

# sampler del config -> (sampler del evaluador, cómo se obtienen las etiquetas)
_SAMPLERS = {"cluster": "cluster", "label_cluster": "cluster", "random_cluster": "cluster", "random": "random", "full": "full"}


def run_cs(cfg: ExperimentConfig, output_dir=None, log=print) -> dict:
    t_start = time.time()
    torch.set_num_threads(cfg.train.num_threads)
    out = Path(output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.json")

    clients, info = build_federated_clients(cfg, seed=cfg.dataset.seed)
    ids = sorted(clients)
    input_shape = tuple(clients[ids[0]].train_dataset[0][0].shape)
    n_classes = info["n_classes"]
    space = CNNSpace(**dataclasses.asdict(cfg.space))
    param_ref = max_n_params(space, input_shape, n_classes)
    class_dist = info["train_partition"].class_distribution
    device, S = cfg.train.device, cfg.sampling

    labels, probe_trainings, quality = None, 0, {}
    if S.sampler == "cluster":
        probe_model = lambda: cnn_to_model(CNNGenome(2, [8, 16], 3, "relu", 0.0, True), input_shape, n_classes)
        delta = probe_deltas(clients, probe_model, cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, cfg.probe.seed, device)
        probe_trainings = len(ids)
        labels = cluster_delta(delta, S.n_clusters, S.method, cfg.seed)
    elif S.sampler == "label_cluster":
        labels = cluster_labels_oracle(class_dist, S.n_clusters, cfg.seed)
    elif S.sampler == "random_cluster":
        labels = cluster_random(len(ids), S.n_clusters, cfg.seed)
    if labels is not None:
        quality = cluster_quality(labels, class_dist, cluster_labels_oracle(class_dist, S.n_clusters, cfg.seed))
        (out / "clusters.json").write_text(json.dumps({"labels": [int(x) for x in labels], "quality": quality}, indent=1))
        log(f"clusters: {quality}")

    ev = ClusterSampleEvaluator(
        clients, input_shape, n_classes, space, dataclasses.asdict(cfg.train),
        {**dataclasses.asdict(cfg.federated), "weight_by_cluster": S.weight_by_cluster}, param_ref,
        labels=labels, sampler=_SAMPLERS[S.sampler], n_reps=S.n_clusters, seed=cfg.seed, device=device, monitor=cfg.monitor, eval_holdout=S.eval_holdout)
    E = cfg.evolution
    hist, rows = run_search(ev, space, E.pop_size, E.n_generations, E.search, E.seed, E.crossover_prob, E.crossover_eta,
                            E.mutation_prob, E.mutation_eta, log=log)
    pd.DataFrame(hist).to_csv(out / "population.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "generation_metrics.csv", index=False)

    final_trainings = 0
    if cfg.final.enabled:
        front = {r["genome"]: r for r in sorted(output_front(hist), key=lambda r: -r["acc_rep"])}  # dedup por genoma
        front = list(front.values())[: cfg.final.max_front]
        frows = []
        for r in front:
            g = CNNGenome(**json.loads(r["genome"]))
            for s in range(cfg.final.n_seeds):
                res = ev.evaluate_test(g, 1000 + s, cfg.final.participation)
                final_trainings += res["n_local_trainings"]
                frows.append({"genome": r["genome"], "init_seed": 1000 + s, "acc_rep": r["acc_rep"],
                              "n_params_norm": r["n_params"] / param_ref, **{f"test_{k}": v for k, v in res.items()}})
        fdf = pd.DataFrame(frows)
        fdf.to_csv(out / "final_front.csv", index=False)
        m = fdf.groupby("genome").agg(a=("test_acc_mean", "mean"), p=("n_params_norm", "first"))
        hv_test = _hv(np.stack([-m.a.values, m.p.values], axis=1))
    else:
        hv_test = float("nan")

    cost = {"probe_local_trainings": probe_trainings, "search_local_trainings": ev.n_local_trainings,
            "search_evals": ev.n_evals, "monitor_node_evals": ev.n_monitor_node_evals,
            "final_local_trainings": final_trainings, "elapsed_sec": time.time() - t_start,
            "n_clients": len(ids), "n_reps_per_eval": len(ev.representatives(0)[0]), "eval_fallback_gen0": int(sum(e == r for e, r in zip(ev.assignment(0)[1], ev.assignment(0)[0]))), "hv_ref": HV_REF.tolist(),
            "hv_test_front": hv_test, "cluster_quality": quality}
    (out / "cost.json").write_text(json.dumps(cost, indent=1))
    return cost
