#!/usr/bin/env python
"""Exploratorio: DBSCAN (con/sin PCA) vs k-means sobre Δ, evaluado con las A ya calculadas (sin reentrenar). Ver docs/TRANSFER_DESIGN.md.

Reconstruye Δ con la misma sonda y semilla que run_transfer_study. Datasets: CIFAR-10 (runs/transfer_study, α en 0.1 0.5 1.0) y FEMNIST (runs/transfer_femnist).
Ej: python run_dbscan_study.py --workers 16
"""
import argparse
import dataclasses
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import run_transfer_study as R
from evofederated.data.federated_split import build_federated_clients
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta, cluster_dbscan, cluster_quality
from evofederated.metrics.transfer import permutation_p, transfer_gap
from evofederated.utils.config import _merge, from_dict, load_config

QS = (0.3, 0.5, 0.7, 0.9)
REPS = {"raw": lambda d: d, "pca95": lambda d: PCA(0.95, svd_solver="full").fit_transform(d), "pca80": lambda d: PCA(0.80, svd_solver="full").fit_transform(d)}
DATASETS = [("cifar10", "configs/cs_cifar10.yaml", "runs/transfer_study", ["0.1", "0.5", "1.0"], range(10)),
            ("femnist", "configs/cs_femnist_natural.yaml", "runs/transfer_femnist", ["natural"], range(20))]


def run_task(args):
    ds, cfgpath, src, alpha, seed, n_perm, out = args
    torch.set_num_threads(2)
    z = np.load(f"{src}/A_alpha{alpha}_seed{seed}.npz")
    A, cd, lab10, oracle = np.nan_to_num(z["A"]), z["cd"], z["lab_delta"], z["lab_oracle"]
    cfg = from_dict(_merge(dataclasses.asdict(load_config(cfgpath)), {"dataset": ({"seed": seed} if alpha == "natural" else {"alpha": float(alpha), "seed": seed})}))
    clients, info = build_federated_clients(cfg, seed=seed)
    ids = sorted(clients)
    shape = tuple(clients[ids[0]].train_dataset[0][0].shape)
    delta = probe_deltas(clients, lambda: R.cnn_to_model(R.PROBE, shape, info["n_classes"]), cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, seed, "cpu")
    C = cfg.sampling.n_clusters
    km10 = cluster_delta(delta, C, "kmeans", seed)
    base = dict(dataset=ds, alpha=alpha, seed=seed, ari_repro=float(cluster_quality(km10, cd, lab10)["ari_vs_label_clusters"]),
                gap_km10=transfer_gap(A, km10), gap_oracle=transfer_gap(A, oracle))
    rows = []
    for rep, f in REPS.items():
        zr = f(delta)
        for q in QS:
            lab, n_noise = cluster_dbscan(zr, q)
            k = int(lab.max() + 1)
            r = dict(base, rep=rep, q=q, k=k, noise_frac=n_noise / len(lab), degenerate=k < 2 or k >= len(lab))
            if not r["degenerate"]:
                u = zr / np.maximum(np.linalg.norm(zr, axis=1, keepdims=True), 1e-12)
                kk = cluster_delta(zr, k, "kmeans", seed)  # misma representación y mismo k: comparación justa
                qual = cluster_quality(lab, cd, oracle)
                r.update(min_size=int(np.bincount(lab).min()), max_size=int(np.bincount(lab).max()), silhouette=float(silhouette_score(u, lab, metric="cosine")),
                         eta2=qual["eta2_label_dist"], ari_oracle=qual["ari_vs_label_clusters"], gap_db=transfer_gap(A, lab), gap_kmk=transfer_gap(A, kk),
                         _lab=lab, _kk=kk)
            rows.append(r)
        ok = [r for r in rows if r["rep"] == rep and not r["degenerate"]]
        for r in ok:
            r["principal"] = False
        if ok:  # variante principal: mayor silueta (criterio no supervisado)
            best = max(ok, key=lambda r: r["silhouette"])
            best["principal"] = True
            best["p_db"] = permutation_p(A, best["_lab"], n_perm, seed)
            best["p_kmk"] = permutation_p(A, best["_kk"], n_perm, seed)
    for r in rows:
        r.pop("_lab", None), r.pop("_kk", None)
    pd.DataFrame(rows).to_csv(f"{out}/part_{ds}_{alpha}_{seed}.csv", index=False)
    return f"{ds} α={alpha} seed={seed} ok"


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out, prev = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, p[i] * (len(p) - rank)))
        out[i] = prev
    return out


def summarize(d, out):
    rng = np.random.RandomState(0)
    ci = lambda x: np.percentile([rng.choice(x, len(x)).mean() for _ in range(5000)], [2.5, 97.5])
    print(f"reproducción k-means C=10: ARI medio {d.groupby(['dataset', 'alpha', 'seed']).ari_repro.first().mean():.4f}")
    print("\n== Sensibilidad por q (todas las seeds; degenerado = k<2) ==")
    print(d.groupby(["dataset", "alpha", "rep", "q"]).agg(degenerate=("degenerate", "sum"), k=("k", "mean"), noise=("noise_frac", "mean"), gap_db=("gap_db", "mean"), gap_kmk=("gap_kmk", "mean")).round(3).to_string())
    pr = d[d.get("principal", False) == True]  # noqa: E712
    cells = []
    for (ds, al, rep), g in pr.groupby(["dataset", "alpha", "rep"]):
        n_tot = len(d[(d.dataset == ds) & (d.alpha == al) & (d.rep == rep)].groupby("seed"))
        r = dict(dataset=ds, alpha=al, rep=rep, n_valid=len(g), n_seeds=n_tot, k=g.k.mean(), noise=g.noise_frac.mean(), eta2=g.eta2.mean(), gap_db=g.gap_db.mean(),
                 gap_kmk=g.gap_kmk.mean(), gap_km10=g.gap_km10.mean(), gap_oracle=g.gap_oracle.mean(), p_lt05_db=int((g.p_db < .05).sum()))
        r["ci_lo"], r["ci_hi"] = ci((g.gap_db).values) if len(g) > 1 else (np.nan, np.nan)
        r["wil_vs_kmk"] = wilcoxon(g.gap_db - g.gap_kmk).pvalue if len(g) > 5 and (g.gap_db != g.gap_kmk).any() else np.nan
        r["wil_vs_km10"] = wilcoxon(g.gap_db - g.gap_km10).pvalue if len(g) > 5 and (g.gap_db != g.gap_km10).any() else np.nan
        cells.append(r)
    s = pd.DataFrame(cells)
    for rep, idx in s.groupby("rep").groups.items():  # Holm sobre las 4 celdas por representación
        for c in ("wil_vs_kmk", "wil_vs_km10"):
            m = s.loc[idx, c].notna()
            s.loc[s.loc[idx].index[m.values], c + "_holm"] = holm(s.loc[idx, c][m].values)
    s.to_csv(Path(out) / "summary.csv", index=False)
    print("\n== Variante principal (mayor silueta) por representación ==")
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--only", help="ds:alpha:seed para un smoke, p. ej. cifar10:0.1:0")
    ap.add_argument("--out", default="runs/dbscan_study")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = [(ds, c, src, al, s, a.n_perm, a.out) for ds, c, src, als, seeds in DATASETS for al in als for s in seeds]
    if a.only:
        ds, al, s = a.only.split(":")
        tasks = [t for t in tasks if (t[0], t[3], t[4]) == (ds, al, int(s))]
    t0 = time.time()
    with Pool(a.workers) as p:
        for msg in p.imap_unordered(run_task, tasks):
            print(msg, f"{time.time() - t0:.0f}s", flush=True)
    d = pd.concat([pd.read_csv(f) for f in sorted(Path(a.out).glob("part_*.csv"))], ignore_index=True)
    d.to_csv(Path(a.out) / "per_seed.csv", index=False)
    summarize(d, a.out)
