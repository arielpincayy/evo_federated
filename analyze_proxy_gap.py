#!/usr/bin/env python
"""Paso A (docs/MOEA_DESIGN.md, "Diagnóstico del cuello de botella"): ¿dónde falla el proxy de 2 representantes frente al entrenamiento con los 100 nodos?
Usa runs/noise_audit/ (accuracy en val de los 100 nodos por arquitectura y sorteo) y runs/arch_node/main/ (accuracy real). Sin entrenar nada."""
import dataclasses

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from evofederated.models.cnn_genome import CNNSpace
from evofederated.utils.config import load_config
from run_arch_node_study import architectures, load as load_truth
from run_noise_audit import scheme_scores

SEEDS, DRAWS, A, REF = (0, 1, 2), 10, 24, 354494
SCHEMES = ("E1_holdout", "E3_todos")


def selection_metrics(P, t):
    """P: (D, A) score del proxy por sorteo; t: (A,) accuracy real. -> dict con τ, recall@k y arrepentimiento (media entre sorteos)."""
    top_half = np.argsort(-t)[: len(t) // 2]
    out = {"tau": np.mean([kendalltau(p, t)[0] for p in P]), "tau_mitad_sup": np.mean([kendalltau(p[top_half], t[top_half])[0] for p in P])}
    for k in (3, 6, 8):
        tk = set(np.argsort(-t)[:k])
        out[f"recall@{k}"] = np.mean([len(tk & set(np.argsort(-p)[:k])) / k for p in P])
    for k in (1, 3, 6):
        out[f"arrepentimiento@{k}(pp)"] = 100 * np.mean([t.max() - t[np.argsort(-p)[:k]].max() for p in P])
    return out


if __name__ == "__main__":
    cfg = load_config("configs/cs_moea.yaml")
    archs = architectures(CNNSpace(**dataclasses.asdict(cfg.space)), A)
    feat = pd.DataFrame(dict(bloques=[len(g.channels) for g in archs], n_fc=[len(g.fc) for g in archs], bn=[int(g.use_bn) for g in archs], dropout=[g.dropout for g in archs],
                             kernel=[g.kernel for g in archs], elu=[int(g.activation == "elu") for g in archs]))
    truth = load_truth("runs/arch_node/main", SEEDS, 2, A)
    rows, bias_rows = [], []
    for s in SEEDS:
        z = [np.load(f"runs/noise_audit/s{s}_d{d}.npz") for d in range(DRAWS)]
        w = z[0]["w"]  # pesos por cluster (tamaño del cluster / N)
        t = ((truth[s]["val"][0].mean(0) + truth[s]["test"][0].mean(0)) / 2).mean(1)  # accuracy real media sobre los 100 nodos, por arquitectura
        size = truth[s]["n_params"] / REF
        for sc in SCHEMES:
            P = np.stack([scheme_scores(zz, s, d)[sc] @ w for d, zz in enumerate(z)])  # (D, A)
            rows.append(dict(seed=s, esquema=sc, **selection_metrics(P, t)))
            bias = P.mean(0) - t  # sesgo del proxy: proxy − real (en accuracy)
            bias_rows.append(dict(seed=s, esquema=sc, sesgo_medio_pp=100 * bias.mean(), sd_sesgo_pp=100 * bias.std(), **{f"rho_sesgo_{c}": spearmanr(bias, feat[c])[0] for c in feat}, rho_sesgo_tamaño=spearmanr(bias, size)[0],
                                  rho_proxy_real_tamaño=spearmanr(P.mean(0) - t.mean() - (t - t.mean()), size)[0]))
        print(f"federación {s}: accuracy real de las 24 arquitecturas: mejor {t.max():.3f}, mediana {np.median(t):.3f}, peor {t.min():.3f}")
    d = pd.DataFrame(rows)
    d.to_csv("runs/proxy_gap.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nCalidad de selección del proxy (media de 3 federaciones; un sorteo; aleatorio ⇒ recall@k ≈ k/24):")
    print(d.groupby("esquema").mean(numeric_only=True).drop(columns="seed").round(3).T.to_string())
    b = pd.DataFrame(bias_rows)
    print("\nSesgo del proxy (proxy − real) y su relación con la arquitectura (Spearman, media de 3 federaciones):")
    print(b.groupby("esquema").mean(numeric_only=True).drop(columns="seed").round(3).T.to_string())
