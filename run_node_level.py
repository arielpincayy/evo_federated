#!/usr/bin/env python
"""Accuracy por nodo de los modelos de cada experimento (reentrena y mide nodo a nodo; ningún experimento previo la guardó).

Trabajos:
  e3     --method M --seed S : por corrida de E3, la población inicial (generación 0), el hall of fame final (frente de Pareto) y su versión
                               refinada (E4); todos entrenados con los 100 nodos (misma receta que la reevaluación final) y medidos en test.
  train                      : arquitectura fija con distintas configuraciones de FedAvg (diagnóstico de entrenamiento), medido en validación.
  noise  --arch I            : arquitectura I del estudio de ruido con C ∈ {10,20,40}, clusters/aleatorio, 2 sorteos, y participación completa (validación).
Salida: runs/node_level/<trabajo>_*.csv en formato largo (una fila por modelo y nodo).
"""
import argparse
import dataclasses
import json
import os

import numpy as np
import pandas as pd
import torch

from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta
from evofederated.federated.fedavg import evaluate_global_on_split
from evofederated.models.cnn_genome import CNNGenome, CNNSpace, cnn_to_model, max_n_params
from evofederated.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True, choices=["e3", "train", "noise"])
ap.add_argument("--method"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--arch", type=int)
ap.add_argument("--n-archs", type=int, default=20)
a = ap.parse_args()
OUT = "runs/node_level"; os.makedirs(OUT, exist_ok=True)


def per_node(model, clients, split):
    r = evaluate_global_on_split(model, clients, split=split)
    return [(cid, v["accuracy"], v["n"]) for cid, v in sorted(r.items())]


def setup(cfg):
    torch.set_num_threads(cfg.train.num_threads)
    clients, info = build_federated_clients(cfg, seed=cfg.dataset.seed)
    shape = tuple(clients[sorted(clients)[0]].train_dataset[0][0].shape); K = info["n_classes"]
    space = CNNSpace(**dataclasses.asdict(cfg.space))
    return clients, shape, K, space, max_n_params(space, shape, K)


def evaluator(cfg, clients, shape, K, space, ref, sampler, labels=None, n_reps=10, fed=None):
    return ClusterSampleEvaluator(clients, shape, K, space, dataclasses.asdict(cfg.train), fed or {**dataclasses.asdict(cfg.federated), "weight_by_cluster": True},
                                  ref, labels=labels, sampler=sampler, n_reps=n_reps, seed=cfg.seed, monitor=False, eval_holdout=True)


rows = []
if a.job == "e3":
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from evofederated.evolution.cs_nas import output_front
    run = f"runs/e3/{a.method}/seed{a.seed}"
    cfg = load_config(f"{run}/config.json")
    clients, shape, K, space, ref = setup(cfg)
    ev = evaluator(cfg, clients, shape, K, space, ref, "full")
    pop = pd.read_csv(f"{run}/population.csv"); ff = pd.read_csv(f"{run}/final_front.csv")
    rc = pd.read_csv(f"{run}/refine/refined_candidates.csv")
    # conjuntos de arquitecturas: población inicial (generación 0), hall of fame = frente de Pareto final (no dominadas por accuracy
    # estimada y parámetros entre todo lo evaluado) y su versión refinada (no dominadas entre los candidatos reevaluados con C=40)
    initial = list(dict.fromkeys(pop[pop.generation == 0].genome))
    hof = list(dict.fromkeys(r["genome"] for r in output_front(pop.to_dict("records"))))
    F = np.array([[-x, p_] for x, p_ in zip(rc.acc_refined, rc.n_params_norm)])
    hof_ref = [rc.genome.iloc[i] for i in NonDominatedSorting().do(F, only_non_dominated_front=True)]
    roles = {}
    for role, gs in (("initial", initial), ("hof", hof), ("hof_refined", hof_ref)):
        for g_ in gs:
            roles.setdefault(g_, []).append(role)
    print(f"{a.method} seed{a.seed}: {len(initial)} iniciales, {len(hof)} en el hall of fame, {len(hof_ref)} refinados, {len(roles)} únicas", flush=True)
    reps = list(ev.valid); w = [1 / len(reps)] * len(reps)
    chk = []
    for k, (gjson, rl) in enumerate(roles.items()):
        fed = ev._train(CNNGenome(**json.loads(gjson)), reps, w, 1000)
        res = per_node(fed["model"], clients, "test")
        for cid, acc, n in res:
            rows.append({"job": "e3", "group": f"{a.method}/seed{a.seed}", "method": a.method, "seed": a.seed, "genome": gjson, "roles": "|".join(rl), "node": cid, "acc": acc, "n": n})
        old = ff[ff.genome == gjson]
        if len(old):  # control de reproducibilidad frente a la reevaluación guardada de E3
            chk.append(abs(np.mean([r[1] for r in res]) - float(old.test_acc_mean.iloc[0])))
        print("  modelo", k + 1, "/", len(roles), flush=True)
    if chk:
        print("control de reproducibilidad: diferencia absoluta media de la accuracy media recalculada vs guardada = %.5f (n=%d)" % (np.mean(chk), len(chk)), flush=True)
    out = f"{OUT}/e3_{a.method}_seed{a.seed}.csv"
elif a.job == "train":
    cfg = load_config("configs/cs_e3.yaml")
    out = f"{OUT}/train_configs.csv"
    for alpha in (0.1, 0.5):
        cfg.dataset.alpha = alpha; cfg.dataset.seed = 0; cfg.seed = 0
        clients, shape, K, space, ref = setup(cfg)
        g = CNNGenome(2, [16, 32], 3, "relu", 0.0, True)
        grid = [("adam1e-3_e2_R5_C10", "adam", 1e-3, 2, 5, 10), ("adam1e-3_e2_R20_C10", "adam", 1e-3, 2, 20, 10), ("adam1e-2_e2_R20_C10", "adam", 1e-2, 2, 20, 10),
                ("adam1e-2_e5_R20_C10", "adam", 1e-2, 5, 20, 10), ("sgd0.05_e2_R20_C10", "sgd", 0.05, 2, 20, 10), ("adam1e-2_e2_R20_full", "adam", 1e-2, 2, 20, 100)]
        for name, opt, lr, ep, rounds, c in grid:
            tr = {**dataclasses.asdict(cfg.train), "optimizer": opt, "lr": lr}
            e = ClusterSampleEvaluator(clients, shape, K, space, tr, {"rounds": rounds, "local_epochs": ep, "weight_by_cluster": True}, ref,
                                       sampler="random" if c < 100 else "full", n_reps=c, seed=0, monitor=False)
            reps, _, w = e.assignment(0)
            fed = e._train(g, reps, w, 0)
            for cid, acc, n in per_node(fed["model"], clients, "val"):
                rows.append({"job": "train", "group": f"alpha{alpha}", "alpha": alpha, "model": name, "node": cid, "acc": acc, "n": n})
            print(alpha, name, "ok", flush=True)
else:  # noise
    cfg = load_config("configs/cs_e3.yaml"); cfg.dataset.seed = 0; cfg.seed = 0; cfg.probe.seed = 0
    clients, shape, K, space, ref = setup(cfg)
    probe = lambda: cnn_to_model(CNNGenome(2, [8, 16], 3, "relu", 0.0, True), shape, K)
    delta = probe_deltas(clients, probe, cfg.probe.k_epochs, cfg.train.lr, cfg.train.optimizer, 0)
    vec = np.random.RandomState(0).rand(a.n_archs, space.n_var)[a.arch]
    g = CNNGenome.from_vector(vec, space)
    out = f"{OUT}/noise_arch{a.arch}.csv"
    for C in (10, 20, 40):
        for fam in ("cluster", "random"):
            e = evaluator(cfg, clients, shape, K, space, ref, fam, labels=cluster_delta(delta, C, "kmeans", 0) if fam == "cluster" else None, n_reps=C)
            for d in range(2):
                reps, _, w = e.assignment(d)
                fed = e._train(g, reps, w, e.init_seed)
                for cid, acc, n in per_node(fed["model"], clients, "val"):
                    rows.append({"job": "noise", "group": f"arch{a.arch}", "arch": a.arch, "model": f"{fam}_C{C}_d{d}", "family": fam, "C": C, "draw": d, "node": cid, "acc": acc, "n": n})
            print(a.arch, fam, C, "ok", flush=True)
    e = evaluator(cfg, clients, shape, K, space, ref, "full")
    reps, _, w = e.assignment(0)
    fed = e._train(g, reps, w, e.init_seed)
    for cid, acc, n in per_node(fed["model"], clients, "val"):
        rows.append({"job": "noise", "group": f"arch{a.arch}", "arch": a.arch, "model": "full", "family": "full", "C": 100, "draw": 0, "node": cid, "acc": acc, "n": n})
pd.DataFrame(rows).to_csv(out, index=False)
print("guardado", out)
