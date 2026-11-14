#!/usr/bin/env python
"""Paso 0 de P3 (informativo, no condiciona la regla): qué k elige la silueta coseno (2..10) en las 20 federaciones de FEMNIST y cómo varía el gap de transferencia con k.
Reusa las A de runs/transfer_femnist. Salida: runs/p3/k_search.csv"""
import dataclasses
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_transfer_study as R
from evofederated.data.federated_split import build_federated_clients
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import choose_k, cluster_delta
from evofederated.metrics.transfer import transfer_gap
from evofederated.utils.config import _merge, from_dict, load_config

CFG = "configs/cs_p3.yaml"


def one(s):
    torch.set_num_threads(2)
    cfg = from_dict(_merge(dataclasses.asdict(load_config(CFG)), {"seed": s, "dataset": {"seed": s}, "probe": {"seed": s}}))
    clients, info = build_federated_clients(cfg, seed=s)
    shape = tuple(clients[0].train_dataset[0][0].shape)
    delta = probe_deltas(clients, lambda: R.cnn_to_model(R.PROBE, shape, info["n_classes"]), cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, cfg.probe.seed, "cpu")
    k_star, sil = choose_k(delta, 10, cfg.seed)
    A = np.nan_to_num(np.load(f"runs/transfer_femnist/A_alphanatural_seed{s}.npz")["A"])
    return [dict(seed=s, k=k, silhouette=sil[k], chosen=k == k_star, gap=transfer_gap(A, cluster_delta(delta, k, "kmeans", cfg.seed))) for k in sorted(sil)]


if __name__ == "__main__":
    Path("runs/p3").mkdir(parents=True, exist_ok=True)
    with Pool(16) as p:
        d = pd.DataFrame([r for rows in p.map(one, range(20)) for r in rows])
    d.to_csv("runs/p3/k_search.csv", index=False)
    print("k elegido por la silueta (nº de semillas):", d[d.chosen].k.value_counts().sort_index().to_dict())
    print(d.groupby("k")[["silhouette", "gap"]].mean().round(4).to_string())
    print("semillas 0-2 (las de P3):", d[d.chosen & (d.seed < 3)][["seed", "k"]].to_dict("records"))
