#!/usr/bin/env python
"""¿Hace falta buscar arquitectura? Con los datos de run_arch_node_study (runs/arch_node/main): dispersión entre arquitecturas y qué tan bien
funciona la elegida con k nodos al evaluarla en el resto (elegir con val de una réplica, evaluar en test de la otra)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_arch_node_study import load

SEEDS, A, KS, DRAWS = [0, 1, 2], 24, (1, 3, 10, 30), 300
D, rng, rows, disp = load("runs/arch_node/main", SEEDS, 2, A), np.random.RandomState(0), [], []
for s in SEEDS:
    (av, _), (at, _) = D[s]["val"], D[s]["test"]
    m = (av.mean(0) + at.mean(0)) / 2  # (A, N)
    pa = 100 * m.mean(1)  # accuracy media por arquitectura (pp)
    disp.append(dict(seed=s, peor=pa.min(), mediana=np.median(pa), mejor=pa.max(), a_menos_2pp_del_mejor=int((pa >= pa.max() - 2).sum()), a_menos_5pp=int((pa >= pa.max() - 5).sum())))
    N = av.shape[2]
    for r, q in ((0, 1), (1, 0)):
        V, T = av[r], at[q]
        for k in KS + (N,):
            out = []
            for _ in range(DRAWS if k < N else 1):
                S = rng.choice(N, k, replace=False) if k < N else np.arange(N)
                rest = np.setdiff1d(np.arange(N), S) if k < N else S
                tr = T[:, rest].mean(1)
                pick = tr[V[:, S].mean(1).argmax()]
                out.append((100 * (pick - tr.mean()), 100 * (tr.max() - tr.mean()), 100 * (pick - tr.max()), (pick - tr.mean()) / (tr.max() - tr.mean())))
            o = np.mean(out, 0)
            rows.append(dict(seed=s, k=k, gana_al_azar_pp=o[0], mejor_posible_pp=o[1], pierde_vs_mejor_pp=o[2], frac_de_la_mejora=o[3]))
        pairs = [rng.choice(N, 2, replace=False) for _ in range(300)]
        rows.append(dict(seed=s, k=-1, spearman_nodo_a_nodo=np.mean([spearmanr(V[:, i], T[:, j]).correlation for i, j in pairs])))
d = pd.DataFrame(rows)
print(pd.DataFrame(disp).round(1).to_string(index=False))
print(d[d.k > 0].groupby("k")[["gana_al_azar_pp", "mejor_posible_pp", "pierde_vs_mejor_pp", "frac_de_la_mejora"]].mean().round(3).to_string())
print("Spearman entre el ranking de arquitecturas de un nodo y el de otro (val vs test, réplicas distintas):", round(d[d.k == -1].spearman_nodo_a_nodo.mean(), 3))
