#!/usr/bin/env python
"""¿El ranking de arquitecturas cambia según el nodo? (docs/ARCH_NODE_DESIGN.md). FEMNIST, modelo global FedAvg evaluado por nodo.

  python run_arch_node_study.py --pilot                      # elige lr en la semilla 100 (3 arquitecturas, lr 1e-3 vs 1e-2)
  python run_arch_node_study.py --lr 0.001 --workers 16      # estudio completo (entrena y analiza)
  python run_arch_node_study.py --lr 0.001 --analyze-only
Salida: runs/arch_node/main/Y_s{semilla}_r{réplica}_a{arq}.npz y, al analizar, per_seed.csv y summary.csv.
"""
import argparse
import dataclasses
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import combine_pvalues, spearmanr

from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.federated.fedavg import evaluate_global_on_split
from evofederated.metrics.arch_node import boot_ci, effects_sd, perm_p, resid_var, selection_gain, stable_interaction
from evofederated.models.cnn_genome import CNNGenome, CNNSpace, max_n_params, n_params
from evofederated.utils.config import load_config

CFG = "configs/cs_femnist_natural.yaml"
INIT_SEEDS = (1000, 1001)  # una por réplica de entrenamiento


def architectures(space, n):
    """n genomas distintos (por estructura, sin contar el dropout) de vectores aleatorios con semilla fija."""
    rng, out, seen = np.random.RandomState(0), [], set()
    while len(out) < n:
        g = CNNGenome.from_vector(rng.rand(space.n_var), space)
        key = (tuple(g.channels), g.kernel, g.activation, g.use_bn, tuple(g.fc), g.pool)
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def run_task(t):
    seed, rep, arch_ids, lr, rounds, epochs, out, threads, n_archs = t
    torch.set_num_threads(threads)
    cfg = load_config(CFG)
    clients, info = build_federated_clients(cfg, seed=seed)
    ids = sorted(clients)
    shape, K = tuple(clients[ids[0]].train_dataset[0][0].shape), info["n_classes"]
    space = CNNSpace(**dataclasses.asdict(cfg.space))
    ev = ClusterSampleEvaluator(clients, shape, K, space, {**dataclasses.asdict(cfg.train), "lr": lr},
                                {"rounds": rounds, "local_epochs": epochs, "weight_by_cluster": True}, max_n_params(space, shape, K),
                                sampler="full", seed=cfg.seed, monitor=False, eval_holdout=True)
    reps = list(ev.valid)
    w = [1.0 / len(reps)] * len(reps)  # participación completa, pesos uniformes
    archs, msgs = architectures(space, n_archs), []
    for a in arch_ids:
        f = Path(out) / f"Y_s{seed}_r{rep}_a{a}.npz"
        if f.exists():
            continue
        t0 = time.time()
        fed = ev._train(archs[a], reps, w, INIT_SEEDS[rep])
        r = {s: evaluate_global_on_split(fed["model"], clients, split=s) for s in ("val", "test")}
        np.savez(f, **{f"{k}_{s}": np.array([r[s][c][k] for c in ids]) for s in r for k in ("accuracy", "n")}, n_params=n_params(fed["model"]))
        msgs.append(f"{f.name} {time.time() - t0:.0f}s")
    return "; ".join(msgs) or "ya existían"


def load(out, seeds, reps, n_archs):
    """-> por semilla: acc[split] (R, A, N) y n[split] (N,)."""
    D = {}
    for s in seeds:
        z = {(r, a): np.load(Path(out) / f"Y_s{s}_r{r}_a{a}.npz") for r in range(reps) for a in range(n_archs)}
        D[s] = {sp: (np.stack([np.stack([z[r, a][f"accuracy_{sp}"] for a in range(n_archs)]) for r in range(reps)]), z[0, 0][f"n_{sp}"]) for sp in ("val", "test")}
        D[s]["n_params"] = np.array([z[0, a]["n_params"] for a in range(n_archs)])
    return D


def logit(acc, n):
    p = (np.round(acc * n) + 0.5) / (n + 1)
    return np.log(p / (1 - p))


def analyze(out, seeds, reps, n_archs, n_perm, n_boot):
    D, rows = load(out, seeds, reps, n_archs), []
    for s in seeds:
        labels = np.load(f"runs/transfer_femnist/A_alphanatural_seed{s}.npz")["lab_delta"]
        for scale in ("acc", "logit"):
            (av, nv), (at, nt) = D[s]["val"], D[s]["test"]
            Yv, Yt = (av, at) if scale == "acc" else (logit(av, nv), logit(at, nt))
            N = Yv.shape[2]
            V, sd_a, sd_n = stable_interaction(Yv, Yt), *effects_sd(Yv, Yt)
            gn, gc = selection_gain(Yv, Yt, labels)
            ci = lambda f: boot_ci(f, N, n_boot, s)
            v_lo, v_hi = ci(lambda i: stable_interaction(Yv[:, :, i], Yt[:, :, i]))
            gn_lo, gn_hi = ci(lambda i: selection_gain(Yv[:, :, i], Yt[:, :, i], labels[i])[0])
            gc_lo, gc_hi = ci(lambda i: selection_gain(Yv[:, :, i], Yt[:, :, i], labels[i])[1])
            sc = 100 if scale == "acc" else 1  # pp para accuracy, unidades de logit para logit
            rows.append(dict(seed=s, scale=scale, unit="pp" if scale == "acc" else "logit", V=V, V_lo=v_lo, V_hi=v_hi, p=perm_p(Yv, Yt, n_perm, s),
                             sd_int=sc * np.sqrt(max(V, 0)), sd_arch=sc * sd_a, sd_node=sc * sd_n, frac_stable=V / resid_var(Yv, Yt),
                             gain_node=sc * gn, gain_node_lo=sc * gn_lo, gain_node_hi=sc * gn_hi, gain_cluster=sc * gc, gain_cluster_lo=sc * gc_lo, gain_cluster_hi=sc * gc_hi,
                             acc_mean=100 * float(np.mean((av.mean() + at.mean()) / 2)),
                             spearman_reps=float(spearmanr(av[0].mean(1), av[1].mean(1)).correlation) if reps > 1 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "per_seed.csv", index=False)
    marc = {s: (D[s]["val"][0].mean((0, 2)) + D[s]["test"][0].mean((0, 2))) / 2 for s in seeds}  # accuracy media por arquitectura y semilla
    rs = [spearmanr(marc[a], marc[b]).correlation for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    summ = []
    for scale, g in df.groupby("scale"):
        summ.append(dict(scale=scale, unit=g.unit.iloc[0], n_seeds=len(g), sd_arch=g.sd_arch.mean(), sd_node=g.sd_node.mean(), sd_int=g.sd_int.mean(), frac_stable=g.frac_stable.mean(),
                         V_ci_pos=int((g.V_lo > 0).sum()), p_fisher=combine_pvalues(g.p, method="fisher")[1], gain_node=g.gain_node.mean(),
                         gain_node_ci_pos=int((g.gain_node_lo > 0).sum()), gain_cluster=g.gain_cluster.mean(), gain_cluster_ci_pos=int((g.gain_cluster_lo > 0).sum()),
                         spearman_reps=g.spearman_reps.mean(), spearman_seeds=float(np.mean(rs)) if rs else np.nan))
    s = pd.DataFrame(summ)
    s.to_csv(Path(out) / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print(s.round(4).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--n_archs", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n_perm", type=int, default=10_000)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/arch_node")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if a.pilot:  # semilla 100 (fuera del estudio), 3 arquitecturas, lr 1e-3 vs 1e-2; regla: mayor accuracy media en val, empate (< 0.5 pp) -> 1e-3
        lrs, tasks = (1e-3, 1e-2), []
        for lr in lrs:
            Path(f"{a.out}/pilot_lr{lr}").mkdir(parents=True, exist_ok=True)
            tasks += [(100, 0, [i], lr, a.rounds, a.local_epochs, f"{a.out}/pilot_lr{lr}", a.threads, a.n_archs) for i in range(3)]
        t0 = time.time()
        with Pool(min(a.workers, len(tasks))) as p:
            for m in p.imap_unordered(run_task, tasks):
                print(m, flush=True)
        acc = {lr: np.mean([np.load(f)["accuracy_val"].mean() for f in Path(f"{a.out}/pilot_lr{lr}").glob("Y_*.npz")]) for lr in lrs}
        print({lr: round(100 * v, 2) for lr, v in acc.items()}, f"piloto {time.time() - t0:.0f}s")
        print("lr elegido:", 1e-2 if acc[1e-2] - acc[1e-3] >= 0.005 else 1e-3)
        raise SystemExit
    out = f"{a.out}/main"
    Path(out).mkdir(parents=True, exist_ok=True)
    if not a.analyze_only:
        assert a.lr, "--lr requerido (sale del piloto)"
        json.dump(vars(a), open(f"{out}/args.json", "w"), indent=1)
        blocks = [list(range(i, min(i + a.chunk, a.n_archs))) for i in range(0, a.n_archs, a.chunk)]
        tasks = [(s, r, b, a.lr, a.rounds, a.local_epochs, out, a.threads, a.n_archs) for s in a.seeds for r in range(a.reps) for b in blocks]
        t0 = time.time()
        with Pool(a.workers) as p:
            for i, m in enumerate(p.imap_unordered(run_task, tasks), 1):
                print(f"[{i}/{len(tasks)} {time.time() - t0:.0f}s] {m}", flush=True)
    analyze(out, a.seeds, a.reps, a.n_archs, a.n_perm, a.n_boot)
