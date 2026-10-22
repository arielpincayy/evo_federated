#!/usr/bin/env python
"""Diagnóstico de entrenamiento: ¿qué configuración de FedAvg con C representantes lleva a una arquitectura fija a una
accuracy razonable? Evalúa el modelo global en el val de TODOS los nodos (solo diagnóstico, no es búsqueda)."""
import argparse
import dataclasses
import itertools
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch

from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.federated.fedavg import evaluate_global_on_split
from evofederated.models.cnn_genome import CNNGenome, CNNSpace, max_n_params
from evofederated.utils.config import load_config


def job(args):
    cfgpath, alpha, name, opt, lr, epochs, bn, rounds, c = args
    torch.set_num_threads(2)
    cfg = load_config(cfgpath)
    cfg.dataset.alpha = alpha
    clients, info = build_federated_clients(cfg, seed=0)
    shape = tuple(clients[0].train_dataset[0][0].shape)
    space = CNNSpace(**dataclasses.asdict(cfg.space))
    g = CNNGenome(2, [16, 32], 3, "relu", 0.0, bn)
    ev = ClusterSampleEvaluator(clients, shape, info["n_classes"], space, {"lr": lr, "optimizer": opt},
                                {"rounds": rounds, "local_epochs": epochs, "weight_by_cluster": True},
                                max_n_params(space, shape, info["n_classes"]), sampler="random" if c < 100 else "full",
                                n_reps=c, seed=0, monitor=False)
    reps, w = ev.assignment(0)[0], ev.assignment(0)[2]
    t0 = time.time()
    fed = ev._train(g, reps, w, 0)
    r = evaluate_global_on_split(fed["model"], clients, split="val")
    a = np.array([v["accuracy"] for v in r.values()]); n = np.array([v["n"] for v in r.values()], float)
    return {"alpha": alpha, "name": name, "opt": opt, "lr": lr, "epochs": epochs, "bn": bn, "rounds": rounds, "C": c,
            "acc_mean": a.mean(), "acc_pooled": (a * n).sum() / n.sum(), "acc_worst": a.min(), "sec": time.time() - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cs_fmnist.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    grid = []
    for alpha in (0.1, 0.5):
        for opt, lr in (("adam", 1e-3), ("adam", 1e-2), ("sgd", 0.05)):
            for epochs, rounds in ((2, 5), (2, 20), (5, 20)):
                grid.append((a.config, alpha, f"{opt}_{lr}", opt, lr, epochs, True, rounds, 10))
        grid.append((a.config, alpha, "adam_0.01_nobn", "adam", 1e-2, 2, False, 20, 10))
        grid.append((a.config, alpha, "adam_0.01_fullpart", "adam", 1e-2, 2, True, 20, 100))  # techo: todos los nodos
    with mp.Pool(a.procs) as p:
        rows = []
        for r in p.imap_unordered(job, grid):
            rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).sort_values(["alpha", "name", "rounds", "epochs"]).to_csv(a.output, index=False)
