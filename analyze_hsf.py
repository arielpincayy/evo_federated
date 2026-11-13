#!/usr/bin/env python
"""Exploratorio (post hoc): ¿agrupar escritores por hsf_id (serie de formularios NIST) transfiere como Δ? Reusa las A de runs/transfer_femnist."""
import dataclasses
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score

from evofederated.data.datasets import load_femnist, natural_partition
from evofederated.data.federated_split import build_federated_clients
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta
from evofederated.metrics.transfer import permutation_p, transfer_gap
from evofederated.utils.config import load_config
import run_transfer_study as R

OUT, N_PERM = "runs/transfer_femnist", 2000
torch.set_num_threads(2)
cfg = load_config("configs/cs_femnist_natural.yaml")
_, y, writers = load_femnist(cfg.dataset.data_root)
t = pq.read_table("data/femnist/train.parquet", columns=["writer_id", "hsf_id"]).to_pandas()
hsf_of = t.groupby("writer_id").hsf_id.first().to_dict()
rows = []
for s in range(20):
    z = np.load(f"{OUT}/A_alphanatural_seed{s}.npz")
    A, cd, lab_d10 = np.nan_to_num(z["A"]), z["cd"], z["lab_delta"]
    part = natural_partition(writers, y, 100, s, cfg.dataset.min_samples_per_client, cfg.dataset.val_ratio, cfg.dataset.test_ratio)
    hsf = np.array([hsf_of[writers[part.client_indices[c][0]]] for c in range(100)])
    clients, _ = build_federated_clients(cfg, seed=s)
    delta = probe_deltas(clients, lambda: R.cnn_to_model(R.PROBE, (1, 28, 28), 62), cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, s, "cpu")
    k = len(np.unique(hsf))
    lab_d_k = cluster_delta(delta, k, "kmeans", s)
    off = ~np.eye(100, dtype=bool)
    H = R.hellinger(cd)
    same = lambda l: (l[:, None] == l[None, :])[off].astype(float)
    X = np.c_[np.ones(off.sum()), same(lab_d10), same(hsf), H[off]]
    b = np.linalg.lstsq(X, np.nan_to_num(z["A"])[off], rcond=None)[0]
    rows.append(dict(seed=s, k_hsf=k, gap_hsf=transfer_gap(A, hsf), p_hsf=permutation_p(A, hsf, N_PERM, s),
                     gap_d10=transfer_gap(A, lab_d10), gap_dk=transfer_gap(A, lab_d_k), p_dk=permutation_p(A, lab_d_k, N_PERM, s),
                     ari_d10_hsf=adjusted_rand_score(hsf, lab_d10), ari_dk_hsf=adjusted_rand_score(hsf, lab_d_k),
                     ari_d10_repro=adjusted_rand_score(lab_d10, cluster_delta(delta, 10, "kmeans", s)),
                     coef_delta=b[1], coef_hsf=b[2], coef_hell=b[3]))
    print(s, flush=True)
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/hsf_comparison.csv", index=False)
rng = np.random.RandomState(0)
ci = lambda x: np.percentile([rng.choice(x, len(x)).mean() for _ in range(10000)], [2.5, 97.5])
for c in ["gap_hsf", "gap_d10", "gap_dk", "ari_d10_hsf", "ari_dk_hsf", "ari_d10_repro", "coef_delta", "coef_hsf"]:
    print(f"{c:14s} media {d[c].mean():.4f}  IC95 [{ci(d[c].values)[0]:.4f}, {ci(d[c].values)[1]:.4f}]")
print("seeds con p<0.05: hsf", (d.p_hsf < .05).sum(), "| Δ(C=k)", (d.p_dk < .05).sum(), "| k hsf", d.k_hsf.unique())
print("gap_dk > gap_hsf en", (d.gap_dk > d.gap_hsf).sum(), "/20  wilcoxon(dk-hsf) bilateral p=%.3g" % wilcoxon(d.gap_dk - d.gap_hsf).pvalue)
print("gap_d10 > gap_hsf en", (d.gap_d10 > d.gap_hsf).sum(), "/20  wilcoxon(d10-hsf) bilateral p=%.3g" % wilcoxon(d.gap_d10 - d.gap_hsf).pvalue)
