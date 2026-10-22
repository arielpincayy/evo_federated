#!/usr/bin/env python
"""Ruido del fitness por *qué nodos entrenan*: una arquitectura por proceso.

Para la arquitectura i de un conjunto fijo se mide, con distintos C y estrategias de muestreo, la accuracy del modelo
global en TODOS los nodos (solo diagnóstico) en R sorteos independientes de los nodos de entrenamiento, y la accuracy con
participación completa (referencia). Salida: runs/noise/arch<i>.csv
"""
import argparse
import dataclasses

import numpy as np
import pandas as pd
import torch

from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta
from evofederated.models.cnn_genome import CNNGenome, CNNSpace, cnn_to_model, max_n_params
from evofederated.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/cs_e3.yaml")
ap.add_argument("--arch", type=int, required=True)
ap.add_argument("--n-archs", type=int, default=20)
ap.add_argument("--draws", type=int, default=4)
ap.add_argument("--cs", type=int, nargs="+", default=[10, 20, 40])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--output", default="runs/noise")
a = ap.parse_args()

cfg = load_config(a.config)
torch.set_num_threads(cfg.train.num_threads)
cfg.dataset.seed = a.seed
clients, info = build_federated_clients(cfg, seed=a.seed)
ids = sorted(clients)
shape = tuple(clients[ids[0]].train_dataset[0][0].shape)
K = info["n_classes"]
space = CNNSpace(**dataclasses.asdict(cfg.space))
ref = max_n_params(space, shape, K)
probe = lambda: cnn_to_model(CNNGenome(2, [8, 16], 3, "relu", 0.0, True), shape, K)
delta = probe_deltas(clients, probe, cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, a.seed)
vec = np.random.RandomState(0).rand(a.n_archs, space.n_var)[a.arch: a.arch + 1]
tr, fed = dataclasses.asdict(cfg.train), {**dataclasses.asdict(cfg.federated), "weight_by_cluster": True}


def ev(sampler, c, labels):
    return ClusterSampleEvaluator(clients, shape, K, space, tr, fed, ref, labels=labels, sampler=sampler, n_reps=c,
                                  seed=a.seed, monitor=True, eval_holdout=True)


rows = []
for c in a.cs:
    arms = {"cluster": ev("cluster", c, cluster_delta(delta, c, "kmeans", a.seed)), "random": ev("random", c, None)}
    for name, e in arms.items():
        for d in range(a.draws):
            e.evaluate_population(vec, d)
            r = e.history[-1]
            rows.append({"arch": a.arch, "arm": f"{name}_C{c}", "C": c, "family": name, "draw": d, "acc_rep": r["acc_rep"],
                         "acc_all": r["mon_acc_mean"], "acc_worst": r["mon_acc_worst"], "n_params_norm": r["n_params_norm"]})
        print(a.arch, name, c, "done", flush=True)
full = ev("full", 100, None)
full.evaluate_population(vec, 0)
r = full.history[-1]
rows.append({"arch": a.arch, "arm": "full", "C": 100, "family": "full", "draw": 0, "acc_rep": r["acc_rep"],
             "acc_all": r["mon_acc_mean"], "acc_worst": r["mon_acc_worst"], "n_params_norm": r["n_params_norm"]})
import os
os.makedirs(a.output, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{a.output}/arch{a.arch}.csv", index=False)
print("saved", a.arch)
