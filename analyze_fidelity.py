#!/usr/bin/env python
"""Paso B (docs/MOEA_DESIGN.md): fidelidad del proxy frente a r representantes por cluster (r = 1 en runs/noise_audit, r > 1 en runs/noise_audit_r{r}).
Compara con la accuracy real (runs/arch_node/main): τ de un sorteo, τ en la mitad superior, recall@k, arrepentimiento y ruido, frente al coste (∝ r), con IC95 bootstrap pareado (remuestreo de arquitecturas)."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from evofederated.models.cnn_genome import CNNSpace
from evofederated.utils.config import load_config

from analyze_proxy_gap import A, SEEDS, selection_metrics
from run_arch_node_study import load as load_truth
from run_noise_audit import e3_scores

RS = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]


def folder(r):
    return "runs/noise_audit" if r == 1 else f"runs/noise_audit_r{r}"


if __name__ == "__main__":
    truth = load_truth("runs/arch_node/main", SEEDS, 2, A)
    t = {s: ((truth[s]["val"][0].mean(0) + truth[s]["test"][0].mean(0)) / 2).mean(1) for s in SEEDS}
    P, noise = {}, {}
    for r in RS:
        for s in SEEDS:
            z = [np.load(f) for f in sorted(Path(folder(r)).glob(f"s{s}_d*.npz"))]
            labels = z[0]["labels"]
            wc = np.array([(labels == c).sum() for c in np.unique(labels)]) / len(labels)
            sc = np.stack([e3_scores(zz) for zz in z])  # (D, A, k)
            P[r, s] = sc @ wc
            noise[r, s] = sc.std(0, ddof=1).mean()
    print(f"{'r':>2} {'coste':>6} {'sorteos':>8} {'ruido':>7} {'tau':>6} {'tau mitad sup.':>15} {'recall@3':>9} {'recall@6':>9} {'arrep.@1(pp)':>13} {'arrep.@3(pp)':>13}")
    for r in RS:
        m = [selection_metrics(P[r, s], t[s]) for s in SEEDS]
        g = lambda k: np.mean([x[k] for x in m])
        print(f"{r:>2} {r:>5}x {len(P[r, SEEDS[0]]):>8} {np.mean([noise[r, s] for s in SEEDS]):7.4f} {g('tau'):6.3f} {g('tau_mitad_sup'):15.3f} {g('recall@3'):9.3f} {g('recall@6'):9.3f} {g('arrepentimiento@1(pp)'):13.2f} {g('arrepentimiento@3(pp)'):13.2f}")
    archs = __import__("run_arch_node_study").architectures(CNNSpace(**dataclasses.asdict(load_config("configs/cs_moea.yaml").space)), A)
    bn, blocks = np.array([int(g.use_bn) for g in archs]), np.array([len(g.channels) for g in archs])
    print("\nSesgo del proxy (proxy − real, accuracy en pp) y su relación con la arquitectura (Spearman, media de las federaciones):")
    print(f"{'r':>2} {'sesgo medio':>12} {'ρ(sesgo, BatchNorm)':>20} {'ρ(sesgo, bloques)':>18} {'ρ(sesgo, tamaño)':>17}")
    for r in RS:
        bias = {s: P[r, s].mean(0) - t[s] for s in SEEDS}
        size = {s: truth[s]["n_params"] for s in SEEDS}
        print(f"{r:>2} {100 * np.mean([bias[s].mean() for s in SEEDS]):12.2f} {np.mean([spearmanr(bias[s], bn)[0] for s in SEEDS]):20.3f} "
              f"{np.mean([spearmanr(bias[s], blocks)[0] for s in SEEDS]):18.3f} {np.mean([spearmanr(bias[s], size[s])[0] for s in SEEDS]):17.3f}")
    rng, B = np.random.RandomState(0), 2000
    tau = lambda r, idx: np.mean([kendalltau(p[idx], t[s][idx])[0] for s in SEEDS for p in P[r, s]])
    boots = [rng.randint(0, A, A) for _ in range(B)]
    base = np.array([tau(1, i) for i in boots])
    print("\nτ de un sorteo frente a r = 1 (diferencia e IC95 bootstrap pareado, remuestreo de arquitecturas):")
    for r in [x for x in RS if x != 1]:
        d = np.array([tau(r, i) for i in boots]) - base
        lo, hi = np.percentile(d, [2.5, 97.5])
        obs = tau(r, np.arange(A)) - tau(1, np.arange(A))
        print(f"  r={r}: {obs:+.3f} [{lo:+.3f}, {hi:+.3f}]  " + ("-> cumple la regla (>= +0.05 con IC > 0)" if r <= 4 and obs >= 0.05 and lo > 0 else ""))
