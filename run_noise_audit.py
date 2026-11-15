#!/usr/bin/env python
"""Auditoría de ruido del fitness por cluster (docs/MOEA_DESIGN.md, sección "Auditoría de ruido"). FEMNIST, espacio v2.

  python run_noise_audit.py --workers 16            # entrena (30 tareas) y analiza
  python run_noise_audit.py --analyze-only
Salida: runs/noise_audit/s{semilla}_d{sorteo}.npz (accuracy en val de los 100 nodos para cada arquitectura) y la tabla por esquema de evaluación.
"""
import argparse
import dataclasses
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

import run_transfer_study as RT
from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import choose_k, cluster_delta
from evofederated.federated.fedavg import evaluate_global_on_split
from evofederated.models.cnn_genome import CNNSpace, cnn_to_model, max_n_params
from evofederated.utils.config import _merge, from_dict, load_config
from run_arch_node_study import architectures, load as load_truth

CFG, SCHEMES = "configs/cs_moea.yaml", ("E1_holdout", "E2_rep", "E3_todos", "E4_cinco")


def run_task(t):
    seed, draw, out, n_archs, rounds = t
    f = Path(out) / f"s{seed}_d{draw}.npz"
    if f.exists():
        return f"{f.name} ya existe"
    torch.set_num_threads(2)
    t0 = time.time()
    cfg = from_dict(_merge(dataclasses.asdict(load_config(CFG)), {"seed": seed, "dataset": {"seed": seed}, "probe": {"seed": seed}, "federated": {"rounds": rounds}}))
    clients, info = build_federated_clients(cfg, seed=seed)
    shape, K = tuple(clients[0].train_dataset[0][0].shape), info["n_classes"]
    delta = probe_deltas(clients, lambda: cnn_to_model(RT.PROBE, shape, K), cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, cfg.probe.seed, "cpu")
    k, _ = choose_k(delta, cfg.sampling.k_max, cfg.seed)
    labels = cluster_delta(delta, k, cfg.sampling.method, cfg.seed)
    space = CNNSpace(**dataclasses.asdict(cfg.space))
    ev = ClusterSampleEvaluator(clients, shape, K, space, dataclasses.asdict(cfg.train), {"rounds": rounds, "local_epochs": cfg.federated.local_epochs, "weight_by_cluster": True},
                                max_n_params(space, shape, K), labels=labels, sampler="cluster", seed=cfg.seed, monitor=False, eval_holdout=True, objectives="per_cluster_size")
    reps, evals, w = ev.assignment(draw)  # el mismo sorteo de la búsqueda (generación = draw)
    rows = []
    for g in architectures(space, n_archs):
        per = evaluate_global_on_split(ev._train(g, reps, w, ev.init_seed)["model"], clients, split="val")  # una sola evaluación de los 100 nodos por modelo
        rows.append([per[c]["accuracy"] for c in sorted(per)])
    acc = np.array(rows)
    np.savez(f, acc=acc, reps=reps, evals=evals, w=w, labels=labels)
    return f"{f.name} {time.time() - t0:.0f}s"


def scheme_scores(z, seed, draw):
    """-> {esquema: (A, k)}: score por arquitectura y cluster."""
    acc, reps, evals, labels = z["acc"], z["reps"], z["evals"], z["labels"]
    out = {s: [] for s in SCHEMES}
    for c, (r, e) in enumerate(zip(reps, evals)):
        others = [m for m in np.where(labels == np.unique(labels)[c])[0] if m != r]
        five = np.random.RandomState(1000 * seed + 10 * draw + c).choice(others, min(5, len(others)), replace=False) if others else [r]
        out["E1_holdout"].append(acc[:, e]); out["E2_rep"].append(acc[:, r])
        out["E3_todos"].append(acc[:, others].mean(1) if others else acc[:, r]); out["E4_cinco"].append(acc[:, five].mean(1))
    return {s: np.stack(v, 1) for s, v in out.items()}


def analyze(out, seeds, draws, n_archs):
    truth = load_truth("runs/arch_node/main", seeds, 2, n_archs)
    res = {s: dict(noise=[], spread=[], tau1=[], taum=[]) for s in SCHEMES}
    for sd in seeds:
        z = [np.load(Path(out) / f"s{sd}_d{d}.npz") for d in range(draws)]
        T = (truth[sd]["val"][0].mean(0) + truth[sd]["test"][0].mean(0)) / 2  # (A, N): accuracia real (100 nodos) por nodo
        labels = z[0]["labels"]
        tc = np.stack([T[:, labels == c].mean(1) for c in np.unique(labels)], 1)  # (A, k)
        S = {s: np.stack([scheme_scores(zz, sd, d)[s] for d, zz in enumerate(z)]) for s in SCHEMES}  # (D, A, k)
        for s in SCHEMES:
            res[s]["noise"].append(S[s].std(0, ddof=1).mean())
            res[s]["spread"].append(S[s].mean(0).std(0, ddof=1).mean())
            res[s]["tau1"] += [kendalltau(S[s][d][:, c], tc[:, c])[0] for d in range(draws) for c in range(tc.shape[1])]
            res[s]["taum"] += [kendalltau(S[s].mean(0)[:, c], tc[:, c])[0] for c in range(tc.shape[1])]
    print(f"{'esquema':12s} {'ruido(SD)':>9s} {'señal(SD)':>9s} {'fiabilidad 1 sorteo':>20s} {'tau 1 sorteo':>13s} {'tau media sorteos':>18s}")
    rows = {}
    for s in SCHEMES:
        nz, sp = np.mean(res[s]["noise"]), np.mean(res[s]["spread"])
        vt = max(sp ** 2 - nz ** 2 / draws, 0.0)
        rows[s] = nz
        print(f"{s:12s} {nz:9.4f} {sp:9.4f} {vt / (vt + nz ** 2):20.3f} {np.mean(res[s]['tau1']):13.3f} {np.mean(res[s]['taum']):18.3f}")
    print(f"\nruido por cambiar los representantes que entrenan (≈ E3): {rows['E3_todos']:.4f}")
    print(f"ruido por elegir el nodo de evaluación (E1): {np.sqrt(max(rows['E1_holdout'] ** 2 - rows['E3_todos'] ** 2, 0)):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--n_archs", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="runs/noise_audit")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not a.analyze_only:
        with Pool(a.workers) as p:
            for m in p.imap_unordered(run_task, [(s, d, a.out, a.n_archs, a.rounds) for s in a.seeds for d in range(a.draws)]):
                print(m, flush=True)
    analyze(a.out, a.seeds, a.draws, a.n_archs)
