#!/usr/bin/env python
"""¿Qué tan cohesionados están los clusters por Δ? Tamaños, cohesión (en Δ y en distribución de etiquetas) y alternativas de agrupación.

Reconstruye Δ con la misma sonda/semillas de E3 (no entrena arquitecturas). Incluye un proxy sin entrenamiento del error del
estimador: con una distribución de etiquetas por nodo g_i como sustituto de lo que determina su accuracy, la varianza exacta de
(a) un representante por cluster con peso |cluster|/N y (b) C nodos al azar, para estimar la media de g sobre los N nodos.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

from evofederated.data.federated_split import build_federated_clients
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta, cluster_labels_oracle, cluster_quality, cluster_random
from evofederated.models.cnn_genome import CNNGenome, cnn_to_model
from evofederated.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/cs_e3.yaml")
ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--output", default="runs/clusters")
a = ap.parse_args()
torch.set_num_threads(4)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def var_stratified(g, lab):
    N, tot = len(g), 0.0
    for c in np.unique(lab):
        m = g[lab == c]; tot += (len(m) / N) ** 2 * ((m - m.mean(0)) ** 2).sum() / len(m)
    return tot


def var_random(g, C):
    N = len(g)
    return (1 - C / N) / C * ((g - g.mean(0)) ** 2).sum() / (N - 1)


def hellinger(p, q):
    return np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum())


rows, size_rows = [], []
for alpha in a.alphas:
    for seed in a.seeds:
        cfg = load_config(a.config); cfg.dataset.alpha = alpha; cfg.dataset.seed = seed; cfg.seed = seed; cfg.probe.seed = seed
        clients, info = build_federated_clients(cfg, seed=seed)
        shape = tuple(clients[0].train_dataset[0][0].shape); K = info["n_classes"]
        cd = info["train_partition"].class_distribution.astype(float); g = cd / cd.sum(1, keepdims=True)
        probe = lambda: cnn_to_model(CNNGenome(2, [8, 16], 3, "relu", 0.0, True), shape, K)
        delta = probe_deltas(clients, probe, cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, seed)
        u = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        H = np.array([[hellinger(g[i], g[j]) for j in range(len(g))] for i in range(len(g))])
        schemes = {}
        for C in (5, 10, 20, 40):
            schemes[f"kmeans_delta_C{C}"] = cluster_delta(delta, C, "kmeans", seed)
            schemes[f"average_delta_C{C}"] = cluster_delta(delta, C, "average", seed)
            schemes[f"label_oracle_C{C}"] = cluster_labels_oracle(cd, C, seed)
            schemes[f"random_C{C}"] = cluster_random(len(g), C, seed)
        if seed in (0, 1, 2, 3, 4) and alpha == 0.1:  # contraste con los clusters guardados de E3
            try:
                saved = np.array(json.load(open(f"runs/e3/cs_nsga/seed{seed}/clusters.json"))["labels"])
                same = len(set(zip(saved, schemes["kmeans_delta_C10"]))) == len(set(saved))
                print(f"alpha={alpha} seed={seed}: clusters reconstruidos coinciden con los de E3: {same}", flush=True)
            except FileNotFoundError:
                pass
        for name, lab in schemes.items():
            C = len(np.unique(lab)); sizes = np.bincount(lab)
            same_pair = lab[:, None] == lab[None, :]; off = ~np.eye(len(lab), dtype=bool)
            within_h = H[same_pair & off].mean() if (same_pair & off).any() else np.nan
            between_h = H[~same_pair].mean()
            sil_delta = silhouette_score(u, lab, metric="cosine") if 1 < C < len(lab) else np.nan
            sil_label = silhouette_score(H, lab, metric="precomputed") if 1 < C < len(lab) else np.nan
            dom = g.argmax(1); purity = np.mean([np.bincount(dom[lab == c]).max() / (lab == c).sum() for c in np.unique(lab)])
            rows.append({"alpha": alpha, "seed": seed, "scheme": name, "C": C, "size_min": sizes.min(), "size_median": np.median(sizes), "size_max": sizes.max(),
                         "singletons": int((sizes == 1).sum()), "gini_sizes": gini(sizes), "eta2": cluster_quality(lab, cd)["eta2_label_dist"],
                         "within_hellinger": within_h, "between_hellinger": between_h, "silhouette_delta": sil_delta, "silhouette_label": sil_label,
                         "purity_dominant_class": purity, "var_ratio_vs_random": var_stratified(g, lab) / var_random(g, C)})
        if alpha == 0.1:
            lab = schemes["kmeans_delta_C10"]; sz = np.bincount(lab)
            for c in np.unique(lab):
                m = lab == c
                size_rows.append({"seed": seed, "cluster": c, "size": int(m.sum()), "within_hellinger": H[np.ix_(m, m)][~np.eye(m.sum(), dtype=bool)].mean() if m.sum() > 1 else np.nan,
                                  "dominant_classes": str(np.argsort(-g[m].mean(0))[:3].tolist())})
        print(f"alpha={alpha} seed={seed} listo", flush=True)
import os
os.makedirs(a.output, exist_ok=True)
d = pd.DataFrame(rows); d.to_csv(f"{a.output}/cluster_metrics.csv", index=False)
pd.DataFrame(size_rows).to_csv(f"{a.output}/cluster_sizes_alpha0.1.csv", index=False)
print("guardado")
