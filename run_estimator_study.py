#!/usr/bin/env python
"""E1+E2: calidad de clusters y del estimador de fitness (CS vs baselines) sobre arquitecturas aleatorias.

Para cada sampler y cada sorteo de representantes se entrena cada arquitectura solo con los representantes y se
compara su accuracy estimada (en val de los reps) con la accuracy del MISMO modelo en todos los nodos.
Mide el error del estimador de evaluación a igual coste (C reps x rondas); no mide el efecto de entrenar solo con reps.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta, cluster_labels_oracle, cluster_quality, cluster_random
from evofederated.models.cnn_genome import CNNGenome, CNNSpace, cnn_to_model, max_n_params
from evofederated.utils.config import _merge, from_dict, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--n-archs", type=int, default=30)
ap.add_argument("--draws", type=int, default=3)
ap.add_argument("--alpha", type=float)
ap.add_argument("--lr", type=float)
ap.add_argument("--rounds", type=int)
ap.add_argument("--epochs", type=int)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])  # semillas de partición/probe
a = ap.parse_args()
cfg0 = load_config(a.config)
torch.set_num_threads(cfg0.train.num_threads)
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
rows, quals = [], []
for seed in a.seeds:
    over = {"seed": seed, "dataset": {"seed": seed}, "federated": {}, "train": {}}
    if a.lr: over["train"]["lr"] = a.lr  # también afecta al probe
    if a.alpha: over["dataset"]["alpha"] = a.alpha
    if a.rounds: over["federated"]["rounds"] = a.rounds
    if a.epochs: over["federated"]["local_epochs"] = a.epochs
    cfg = from_dict(_merge(dataclasses.asdict(cfg0), over))
    clients, info = build_federated_clients(cfg, seed=seed)
    ids = sorted(clients)
    shape = tuple(clients[ids[0]].train_dataset[0][0].shape)
    K, C = info["n_classes"], cfg.sampling.n_clusters
    space = CNNSpace(**dataclasses.asdict(cfg.space))
    cd = info["train_partition"].class_distribution
    probe = lambda: cnn_to_model(CNNGenome(2, [8, 16], 3, "relu", 0.0, True), shape, K)
    delta = probe_deltas(clients, probe, cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, cfg.probe.seed)
    oracle = cluster_labels_oracle(cd, C, seed)
    lab_delta = cluster_delta(delta, C, cfg.sampling.method, seed)
    # (sampler, etiquetas, evaluar en otro nodo del cluster que no entrenó)
    arms = {"cluster_delta": ("cluster", lab_delta, True),
            "cluster_label_oracle": ("cluster", oracle, True),
            "cluster_random": ("cluster", cluster_random(len(ids), C, seed), True),
            "random_nodes": ("random", None, True),
            "cluster_delta_noholdout": ("cluster", lab_delta, False),
            "random_nodes_noholdout": ("random", None, False)}
    X = np.random.RandomState(seed).rand(a.n_archs, space.n_var)
    ref = max_n_params(space, shape, K)
    for name, (sampler, lab, holdout) in arms.items():
        if lab is not None and holdout:
            quals.append({"seed": seed, "arm": name, **cluster_quality(lab, cd, oracle)})
        ev = ClusterSampleEvaluator(clients, shape, K, space, dataclasses.asdict(cfg.train),
                                    {**dataclasses.asdict(cfg.federated), "weight_by_cluster": cfg.sampling.weight_by_cluster},
                                    ref, labels=lab, sampler=sampler, n_reps=C, seed=seed, device=cfg.train.device, eval_holdout=holdout)
        for d in range(a.draws):
            ev.evaluate_population(X, d)
        h = pd.DataFrame(ev.history).assign(arm=name, seed=seed)
        rows.append(h[["arm", "seed", "generation", "individual_idx", "acc_rep", "mon_acc_mean", "mon_acc_worst", "n_local_trainings"]])
        print(name, seed, "done", flush=True)
df = pd.concat(rows)
df.to_csv(out / "estimator_raw.csv", index=False)
pd.DataFrame(quals).to_csv(out / "cluster_quality.csv", index=False)
df["err"] = df.acc_rep - df.mon_acc_mean
summ = df.groupby(["arm", "seed"]).apply(lambda g: pd.Series({
    "bias": g.err.mean(), "rmse": np.sqrt((g.err ** 2).mean()),
    "kendall_within_draw": np.nanmean([kendalltau(x.acc_rep, x.mon_acc_mean)[0] for _, x in g.groupby("generation")])})).reset_index()
summ.to_csv(out / "estimator_summary.csv", index=False)
print(summ.groupby("arm")[["bias", "rmse", "kendall_within_draw"]].agg(["mean", "std"]).round(4).to_string())
