#!/usr/bin/env python
"""Estudio PCA vs Δ crudo antes del clustering (solo calidad de clusters, sin NAS).

Por (alpha, seed): una sola sonda -> Δ; mismas filas, distintas representaciones; k-means esférico C fijo.
Salida: runs/pca_study/metrics.csv (una fila por alpha x seed x variante).
Ej: python run_pca_study.py --seeds 0 1 2 3 4 --alphas 0.1 0.5
"""
import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from evofederated.data.federated_split import build_federated_clients
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta, cluster_labels_oracle, cluster_quality
from evofederated.models.cnn_genome import CNNGenome, cnn_to_model
from evofederated.utils.config import from_dict, load_config, _merge

# variante -> función Δ (N x P) -> representación (N x d). Se agrupa con el mismo cluster_delta (coseno).
VARIANTS = {
    "raw": lambda d: d,
    "centered": lambda d: d - d.mean(axis=0),  # aísla el efecto del centrado del de la reducción
    "pca_var80": lambda d: PCA(0.80, svd_solver="full").fit_transform(d),
    "pca_var95": lambda d: PCA(0.95, svd_solver="full").fit_transform(d),
    "pca_k5": lambda d: PCA(5).fit_transform(d),
    "pca_k20": lambda d: PCA(20).fit_transform(d),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cs_cifar10.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5])
    ap.add_argument("--out", default="runs/pca_study")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = dataclasses.asdict(load_config(a.config))
    rows = []
    for alpha in a.alphas:
        for seed in a.seeds:
            cfg = from_dict(_merge(base, {"dataset": {"alpha": alpha, "seed": seed}}))
            t0 = time.time()
            clients, info = build_federated_clients(cfg, seed=seed)
            ids = sorted(clients)
            shape = tuple(clients[ids[0]].train_dataset[0][0].shape)
            nc = info["n_classes"]
            model = lambda: cnn_to_model(CNNGenome(2, [8, 16], 3, "relu", 0.0, True), shape, nc)
            delta = probe_deltas(clients, model, cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, seed, cfg.train.device)
            t_probe = time.time() - t0
            cd = info["train_partition"].class_distribution
            C = cfg.sampling.n_clusters
            oracle = cluster_labels_oracle(cd, C, seed)
            raw = cluster_delta(delta, C, "kmeans", seed)
            for name, f in VARIANTS.items():
                z = f(delta)
                lab = cluster_delta(z, C, "kmeans", seed)
                q = cluster_quality(lab, cd, oracle)
                rows.append(dict(alpha=alpha, seed=seed, variant=name, dim=z.shape[1], **q,
                                 ari_vs_raw=adjusted_rand_score(raw, lab), min_size=int(np.bincount(lab).min())))
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            print(f"alpha={alpha} seed={seed} probe={t_probe:.0f}s P={delta.shape[1]}", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["alpha", "variant"])[["eta2_label_dist", "ari_vs_label_clusters", "ari_vs_raw"]].agg(["mean", "std"]).round(3).to_string())
