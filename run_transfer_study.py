#!/usr/bin/env python
"""Estudio de transferencia entre nodos (ver docs/TRANSFER_DESIGN.md).

Por (alpha, semilla): A[s, j] = accuracy en val del nodo j de un modelo entrenado solo en el nodo s.
Guarda runs/transfer_study/A_alpha{a}_seed{s}.npz (A + etiquetas Δ/oráculo/aleatorio) y, al final, analiza todo.
Ej: python run_transfer_study.py --seeds 0 1 2 3 4 5 6 7 8 9 --alphas 0.1 0.5 1.0 --epochs 5 --workers 12
"""
import argparse
import copy
import dataclasses
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from evofederated.data.federated_split import build_federated_clients
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta, cluster_labels_oracle, cluster_random
from evofederated.metrics.transfer import permutation_p, transfer_gap
from evofederated.models.cnn_genome import CNNGenome, cnn_to_model
from evofederated.utils.config import _merge, from_dict, load_config

PROBE = CNNGenome(2, [8, 16], 3, "relu", 0.0, True)
MEASURE = CNNGenome(2, [16, 32], 3, "relu", 0.0, True)  # distinta de la sonda (evita circularidad)


def run_one(args):
    cfgpath, alpha, seed, epochs, out, threads = args
    f = Path(out) / f"A_alpha{alpha}_seed{seed}.npz"
    if f.exists():
        return f"{f.name} ya existe"
    torch.set_num_threads(threads)
    t0 = time.time()
    cfg = from_dict(_merge(dataclasses.asdict(load_config(cfgpath)), {"dataset": {"alpha": alpha, "seed": seed}}))
    clients, info = build_federated_clients(cfg, seed=seed)
    ids = sorted(clients)
    shape = tuple(clients[ids[0]].train_dataset[0][0].shape)
    nc = info["n_classes"]
    C = cfg.sampling.n_clusters
    cd = info["train_partition"].class_distribution
    delta = probe_deltas(clients, lambda: cnn_to_model(PROBE, shape, nc), cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, seed, cfg.train.device)
    labels = dict(delta=cluster_delta(delta, C, "kmeans", seed), oracle=cluster_labels_oracle(cd, C, seed), random=cluster_random(len(ids), C, seed))

    torch.manual_seed(seed + 1000)
    init = cnn_to_model(MEASURE, shape, nc)  # misma init para todas las fuentes
    A = np.full((len(ids), len(ids)), np.nan)
    for s in ids:
        m = copy.deepcopy(init)
        clients[s].train(m, epochs=epochs, lr=cfg.train.lr, optimizer_name=cfg.train.optimizer, device=cfg.train.device)
        for j in ids:
            if j != s:
                A[s, j] = clients[j].evaluate(m, cfg.train.device, split="val")["accuracy"]
    np.savez(f, A=A, cd=cd, **{f"lab_{k}": v for k, v in labels.items()})
    return f"{f.name} {time.time() - t0:.0f}s"


def hellinger(cd):
    p = cd / np.maximum(cd.sum(1, keepdims=True), 1)
    r = np.sqrt(p)
    return np.sqrt(0.5 * ((r[:, None, :] - r[None, :, :]) ** 2).sum(-1))


def analyze(out, alphas, seeds, n_perm):
    rows = []
    for a in alphas:
        for s in seeds:
            z = np.load(Path(out) / f"A_alpha{a}_seed{s}.npz")
            A, H = z["A"], hellinger(z["cd"])
            r = dict(alpha=a, seed=s, acc_mean=float(np.nanmean(A)))
            for k in ("delta", "oracle", "random"):
                lab = z[f"lab_{k}"]
                r[f"gap_{k}"] = transfer_gap(np.nan_to_num(A), lab)
                if k != "random":
                    r[f"p_{k}"] = permutation_p(A, lab, n_perm, s)
            # exploratorio: A ~ mismo_cluster(Δ) + Hellinger; coeficiente de mismo_cluster
            off = ~np.eye(len(A), dtype=bool)
            same = (z["lab_delta"][:, None] == z["lab_delta"][None, :])[off].astype(float)
            X = np.c_[np.ones(same.size), same, H[off]]
            r["coef_same_given_hellinger"] = float(np.linalg.lstsq(X, A[off], rcond=None)[0][1])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "per_seed.csv", index=False)
    rng = np.random.RandomState(0)
    summ = []
    for a, g in df.groupby("alpha"):
        d = g.gap_delta.values
        boot = [rng.choice(d, len(d)).mean() for _ in range(10_000)]
        w = wilcoxon(g.gap_delta - g.gap_random, alternative="greater")
        summ.append(dict(alpha=a, n=len(g), acc_mean=g.acc_mean.mean(), gap_delta=d.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                         gap_oracle=g.gap_oracle.mean(), gap_random=g.gap_random.mean(), frac_oracle_median=(g.gap_delta / g.gap_oracle).median(),
                         seeds_p_lt_05=int((g.p_delta < 0.05).sum()), wilcoxon_p=w.pvalue, coef_same_given_hellinger=g.coef_same_given_hellinger.mean()))
    s = pd.DataFrame(summ)
    order = s.wilcoxon_p.argsort().values  # Holm
    holm, prev = np.empty(len(s)), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, s.wilcoxon_p.iloc[i] * (len(s) - rank)))
        holm[i] = prev
    s["holm_p"] = holm
    s.to_csv(Path(out) / "summary.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cs_cifar10.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n_perm", type=int, default=10_000)
    ap.add_argument("--out", default="runs/transfer_study")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not a.analyze_only:
        tasks = [(a.config, al, s, a.epochs, a.out, a.threads) for al in a.alphas for s in a.seeds]
        with Pool(a.workers) as p:
            for msg in p.imap_unordered(run_one, tasks):
                print(msg, flush=True)
    analyze(a.out, a.alphas, a.seeds, a.n_perm)
