#!/usr/bin/env python
"""Paso 0 del v4 (docs/MOEA_DESIGN.md, "MOEA v4"): piloto de un espacio de arquitecturas. N arquitecturas aleatorias del espacio de la config, federación 0,
proxy del v3 (2 representantes por cluster, 20 rondas x 2 épocas; score = media del cluster sobre los miembros que no entrenaron). Mide tiempo por evaluación,
si las arquitecturas aprenden y qué tan dispersas son, y sugiere `eval_budget`.

  python run_space_pilot.py --config configs/cs_moea4.yaml --n 60 --workers 16
Salida: runs/space_pilot/<nombre>/chunk*.csv y el resumen por pantalla."""
import argparse
import dataclasses
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_transfer_study as RT
from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import choose_k, cluster_delta
from evofederated.models.cnn_genome import CNNSpace, cnn_to_model, max_n_params, n_params
from evofederated.utils.config import load_config
from run_arch_node_study import architectures


def run_chunk(t):
    config, idx, n, out, rounds, seed = t
    torch.set_num_threads(2)
    cfg = load_config(config)
    cfg.dataset.seed, cfg.seed, cfg.probe.seed = seed, seed, seed
    if rounds:
        cfg.federated.rounds = rounds
    clients, info = build_federated_clients(cfg, seed=seed)
    shape, K = tuple(clients[0].train_dataset[0][0].shape), info["n_classes"]
    delta = probe_deltas(clients, lambda: cnn_to_model(RT.PROBE, shape, K), cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, cfg.probe.seed, "cpu")
    k, _ = choose_k(delta, cfg.sampling.k_max, cfg.seed)
    labels = cluster_delta(delta, k, cfg.sampling.method, cfg.seed)
    space = CNNSpace(**dataclasses.asdict(cfg.space))
    ev = ClusterSampleEvaluator(clients, shape, K, space, dataclasses.asdict(cfg.train), {**dataclasses.asdict(cfg.federated), "weight_by_cluster": cfg.sampling.weight_by_cluster},
                                max_n_params(space, shape, K), labels=labels, sampler="cluster", seed=cfg.seed, monitor=False, eval_holdout=cfg.sampling.eval_holdout,
                                objectives="per_cluster_mean_size", reps_per_cluster=cfg.sampling.reps_per_cluster)
    reps, _, w = ev.assignment(0)
    cw, archs, rows = ev._cluster_w(reps, w), architectures(space, n), []
    for i in idx:
        t0 = time.time()
        fed = ev._train(archs[i], reps, w, ev.init_seed)
        obj = ev._cluster_mean_acc(fed["model"], reps)
        rows.append(dict(arch=i, time_s=time.time() - t0, acc=float(np.dot(cw, obj)), size=n_params(fed["model"]) / ev.param_ref, k=k, blocks=len(archs[i].channels), n_fc=len(archs[i].fc),
                         kernel=archs[i].kernel, act=archs[i].activation, bn=int(archs[i].use_bn)))
    pd.DataFrame(rows).to_csv(Path(out) / f"chunk{idx[0]}.csv", index=False)
    return f"chunk{idx[0]}: {len(rows)} arquitecturas"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=0, help="solo pruebas: menos rondas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"runs/space_pilot/{Path(a.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    chunks = [list(c) for c in np.array_split(np.arange(a.n), min(a.workers, a.n))]
    with Pool(len(chunks)) as p:
        for m in p.imap_unordered(run_chunk, [(a.config, c, a.n, str(out), a.rounds, a.seed) for c in chunks]):
            print(m, flush=True)
    d = pd.concat([pd.read_csv(f) for f in sorted(out.glob("chunk*.csv"))]).sort_values("arch").reset_index(drop=True)
    d.to_csv(out / "pilot.csv", index=False)
    med = d.time_s.median()
    budget = int(min(980, max(420, 7200 / (1.3 * med))) // 28 * 28)
    print(f"\nespacio {Path(a.config).stem}: {len(d)} arquitecturas, k={int(d.k.iloc[0])}")
    print(f"tiempo por evaluación: mediana {med:.1f} s, p90 {d.time_s.quantile(.9):.1f} s, máx {d.time_s.max():.1f} s (con {len(chunks)} procesos en paralelo)")
    print(f"accuracy del proxy: media {d.acc.mean():.3f}, SD {d.acc.std():.3f}, mejor {d.acc.max():.3f}; aprenden (> 0.05): {100 * (d.acc > 0.05).mean():.0f} %")
    print(f"tamaño normalizado: mediana {d['size'].median():.3f}, máx {d['size'].max():.3f}")
    for m in (28, len(d)):
        s = d.head(m)
        print(f"fracción a <= 0.05 de la mejor entre las primeras {m}: {100 * (s.acc >= s.acc.max() - 0.05).mean():.0f} %")
    print(f"REGLA: mediana <= 20 s -> {'sí' if med <= 20 else 'NO'}; aprenden >= 70 % -> {'sí' if (d.acc > 0.05).mean() >= 0.7 else 'NO'}")
    print(f"eval_budget sugerido = {budget}")
    pops = [pd.read_csv(f"runs/moea2/rs/seed{s}/population.csv") for s in range(10)]
    ref = np.mean([(g.acc_rep >= g.acc_rep.max() - 0.05).mean() for g in (p[p.generation == 0] for p in pops)])
    print(f"(referencia, espacio actual: fracción a <= 0.05 de la mejor entre las 28 de la generación 0 del rs del v2: {100 * ref:.0f} %)")
