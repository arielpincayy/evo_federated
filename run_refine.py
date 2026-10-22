#!/usr/bin/env python
"""Refinamiento multifidelidad de una corrida existente (runs/e3/<método>/seed<s>):

1. candidatos = capas no dominadas del frente de salida (≤ max-cand arquitecturas, por accuracy estimada media);
2. se reevalúan con C=refine-c nodos aleatorios nuevos (fidelidad mayor, otra semilla de nodos);
3. el frente final se elige con esa estimación y se reentrena con participación completa para medir en test.
El mismo procedimiento se aplica a todos los métodos, de modo que las comparaciones entre métodos siguen siendo a igual coste.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.evolution.cs_nas import _hv
from evofederated.evolution.refine import candidate_set, select_final
from evofederated.models.cnn_genome import CNNGenome, CNNSpace, max_n_params
from evofederated.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--refine-c", type=int, default=40)
ap.add_argument("--max-cand", type=int, default=12)
ap.add_argument("--final-k", type=int, default=5)
ap.add_argument("--participation", default="full")
ap.add_argument("--output")
a = ap.parse_args()

t0 = time.time()
run = Path(a.run)
cfg = load_config(run / "config.json")
torch.set_num_threads(cfg.train.num_threads)
clients, info = build_federated_clients(cfg, seed=cfg.dataset.seed)
shape = tuple(clients[sorted(clients)[0]].train_dataset[0][0].shape)
K = info["n_classes"]
space = CNNSpace(**dataclasses.asdict(cfg.space))
ref = max_n_params(space, shape, K)
ev = ClusterSampleEvaluator(clients, shape, K, space, dataclasses.asdict(cfg.train),
                            {**dataclasses.asdict(cfg.federated), "weight_by_cluster": True}, ref, sampler="random",
                            n_reps=a.refine_c, seed=cfg.seed, monitor=True, eval_holdout=True)

pop = pd.read_csv(run / "population.csv").to_dict("records")
cands = candidate_set(pop, a.max_cand)
refined = []
for c in cands:
    ev.evaluate_population(np.array([c["vector"]]), generation=1000)  # reps nuevos, independientes de la búsqueda
    r = ev.history[-1]
    refined.append({"genome": c["genome"], "n_params_norm": c["n_params_norm"], "n_evals_search": c["n_evals"],
                    "acc_rep_search": c["acc_rep_mean"], "acc_refined": r["acc_rep"], "acc_all_refined": r["mon_acc_mean"]})
    print("refinado", len(refined), "/", len(cands), flush=True)
refine_trainings = ev.n_local_trainings

final = select_final(refined, a.final_k)
rows, final_trainings = [], 0
for r in final:
    res = ev.evaluate_test(CNNGenome(**json.loads(r["genome"])), 1000, a.participation)
    final_trainings += res["n_local_trainings"]
    rows.append({**r, **{f"test_{k}": v for k, v in res.items()}})
df = pd.DataFrame(rows)
out = Path(a.output or run / "refine")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "refined_final_front.csv", index=False)
pd.DataFrame(refined).to_csv(out / "refined_candidates.csv", index=False)
m = df.groupby("genome").agg(a=("test_acc_mean", "mean"), p=("n_params_norm", "first"))
prev = json.load(open(run / "cost.json"))
cost = {"n_candidates": len(cands), "refine_c": a.refine_c, "refine_local_trainings": refine_trainings,
        "final_local_trainings": final_trainings, "hv_test_refined": _hv(np.stack([-m.a.values, m.p.values], axis=1)),
        "hv_test_original": prev["hv_test_front"], "search_local_trainings": prev["search_local_trainings"],
        "elapsed_sec": time.time() - t0}
(out / "refine_cost.json").write_text(json.dumps(cost, indent=1))
print(cost)
