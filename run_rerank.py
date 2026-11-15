#!/usr/bin/env python
"""Paso C (docs/MOEA_DESIGN.md, "Diagnóstico del cuello de botella"): ¿el archivo del MOEA contiene mejores candidatos que el de la aleatoria?
Sin nueva búsqueda: de cada archivo (runs/moea2/{brazo}/seed{s}/population.csv) se eligen M candidatos con select_hof y se reentrenan con los 100 nodos
(mismo protocolo que la etapa final: 20 rondas x 2 épocas, init 1000, test). Los ya reentrenados en final_front.csv se reutilizan.

  python run_rerank.py --workers 16        # reentrena y analiza
  python run_rerank.py --analyze-only
Salida: runs/rerank/{brazo}_seed{s}.csv (una fila por candidato) y la tabla de contrastes."""
import argparse
import dataclasses
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from evofederated.data.federated_split import build_federated_clients
from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.evolution.cs_nas import _hv, _nd
from evofederated.evolution.refine import select_hof
from evofederated.models.cnn_genome import CNNGenome, CNNSpace, max_n_params
from evofederated.utils.config import load_config

ARMS = ("nsga3", "rs")


def pick_candidates(records, M, k, prefix=0):
    """M candidatos con select_hof sobre los primeros `prefix` registros del archivo (todos si prefix = 0)."""
    return select_hof(records[:prefix] if prefix else records, K=M, n_acc=k)


def _setup(root, arm, seed, rounds, threads):
    """Reconstruye el evaluador de una corrida guardada. -> (cfg, evaluador, param_ref, clusters.json, carpeta)."""
    torch.set_num_threads(threads)
    d = Path(root) / arm / f"seed{seed}"
    cfg = load_config(d / "config.json")
    if rounds:  # solo para pruebas rápidas
        cfg.federated.rounds = rounds
    cl = json.load(open(d / "clusters.json"))
    clients, info = build_federated_clients(cfg, seed=cfg.dataset.seed)
    ids = sorted(clients)
    shape, K = tuple(clients[ids[0]].train_dataset[0][0].shape), info["n_classes"]
    space = CNNSpace(**dataclasses.asdict(cfg.space))
    ref = max_n_params(space, shape, K)
    ev = ClusterSampleEvaluator(clients, shape, K, space, dataclasses.asdict(cfg.train), {**dataclasses.asdict(cfg.federated), "weight_by_cluster": cfg.sampling.weight_by_cluster}, ref,
                                labels=np.array(cl["labels"]), sampler="cluster", n_reps=cfg.sampling.n_clusters, seed=cfg.seed, monitor=False, eval_holdout=cfg.sampling.eval_holdout,
                                objectives=cfg.evolution.objectives, size_penalty=cfg.evolution.size_penalty, cache=not cfg.evolution.reeval_parents)
    return cfg, ev, ref, cl, d


def run_task(t):
    root, arm, seed, M, out, threads, rounds = t
    f = Path(out) / f"{arm}_seed{seed}.csv"
    if f.exists():
        return f"{f.name} ya existe"
    t0 = time.time()
    cfg, ev, ref, cl, d = _setup(root, arm, seed, rounds, threads)
    cand = pick_candidates(pd.read_csv(d / "population.csv").to_dict("records"), M, cl["k"])
    ff = pd.read_csv(d / "final_front.csv")
    known = {r.genome: (r.test_acc_mean, r.n_params_norm) for r in ff[ff.roles.str.split("|").map(lambda x: "hof" in x)].itertuples()}
    part, rows = Path(str(f) + ".partial"), []
    if part.exists():
        rows = [json.loads(x) for x in part.read_text().splitlines()]
    done = {r["genome"] for r in rows}
    for c in cand:
        if c["genome"] in done:
            continue
        if c["genome"] in known and not rounds:
            acc, size, reused = known[c["genome"]][0], known[c["genome"]][1], True
        else:
            res = ev.evaluate_test(CNNGenome(**json.loads(c["genome"])), 1000, cfg.final.participation)
            acc, size, reused = res["acc_mean"], res["n_params"] / ref, False
        rows.append(dict(arm=arm, seed=seed, genome=c["genome"], acc=float(acc), size=float(size), reused=reused, reason=c["reason"]))
        with open(part, "a") as fh:
            fh.write(json.dumps(rows[-1]) + "\n")
    pd.DataFrame(rows).to_csv(f, index=False)
    part.unlink(missing_ok=True)
    return f"{f.name} {len(rows)} candidatos ({sum(not r['reused'] for r in rows)} reentrenados) {time.time() - t0:.0f}s"


def run_prefix_task(t):
    """Paso D: reentrena los M candidatos que se elegirían tras solo las primeras B evaluaciones, para cada B (comparte caché entre presupuestos)."""
    root, arm, seed, budgets, M, out, threads, rounds, rerank_dir = t
    todo = [B for B in budgets if not (Path(out) / f"{arm}_seed{seed}_B{B}.csv").exists()]
    if not todo:
        return f"{arm}_seed{seed} ya existe"
    t0 = time.time()
    cfg, ev, ref, cl, d = _setup(root, arm, seed, rounds, threads)
    records = pd.read_csv(d / "population.csv").to_dict("records")
    known = {}
    if not rounds:  # solo se reutilizan reentrenamientos con el protocolo completo
        ff = pd.read_csv(d / "final_front.csv")
        known.update({r.genome: (r.test_acc_mean, r.n_params_norm) for r in ff[ff.roles.str.split("|").map(lambda x: "hof" in x)].itertuples()})
        rr = Path(rerank_dir) / f"{arm}_seed{seed}.csv"
        if rr.exists():
            known.update({r.genome: (r.acc, r.size) for r in pd.read_csv(rr).itertuples()})
    cache = Path(out) / f"{arm}_seed{seed}.cache.jsonl"  # reentrenamientos propios de esta corrida (reanudable)
    if cache.exists():
        known.update({j["genome"]: (j["acc"], j["size"]) for j in map(json.loads, cache.read_text().splitlines())})
    n_new = 0
    for B in todo:
        rows = []
        for c in pick_candidates(records, M, cl["k"], B):
            if c["genome"] not in known:
                res = ev.evaluate_test(CNNGenome(**json.loads(c["genome"])), 1000, cfg.final.participation)
                known[c["genome"]] = (res["acc_mean"], res["n_params"] / ref)
                n_new += 1
                with open(cache, "a") as fh:
                    fh.write(json.dumps(dict(genome=c["genome"], acc=float(known[c["genome"]][0]), size=float(known[c["genome"]][1]))) + "\n")
            rows.append(dict(arm=arm, seed=seed, budget=B, genome=c["genome"], acc=float(known[c["genome"]][0]), size=float(known[c["genome"]][1]), reason=c["reason"]))
        pd.DataFrame(rows).to_csv(Path(out) / f"{arm}_seed{seed}_B{B}.csv", index=False)
    return f"{arm}_seed{seed}: presupuestos {todo}, {n_new} reentrenados {time.time() - t0:.0f}s"


def holm(p):
    p = np.asarray(p, float)
    out, prev = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        prev = max(prev, min(1.0, p[i] * (len(p) - rank)))
        out[i] = prev
    return out


def analyze_prefix(out, seeds, budgets, root):
    hv = lambda g: _hv(np.stack([-g.acc.values, g["size"].values], 1))
    rows = []
    for arm in ARMS:
        for s in seeds:
            for B in budgets:
                f = Path(out) / f"{arm}_seed{s}_B{B}.csv"
                if f.exists():
                    rows.append(dict(arm=arm, seed=s, budget=B, hv=hv(pd.read_csv(f))))
            ff = pd.read_csv(Path(root) / arm / f"seed{s}" / "final_front.csv")
            ff = ff[ff.roles.str.split("|").map(lambda x: "hof" in x)]
            rows.append(dict(arm=arm, seed=s, budget=int(json.load(open(Path(root) / arm / f"seed{s}" / "cost.json"))["search_evals"]),
                             hv=_hv(np.stack([-ff.test_acc_mean.values, ff.n_params_norm.values], 1))))  # presupuesto completo: el hall of fame original
    d = pd.DataFrame(rows)
    d.to_csv(Path(out) / "per_run.csv", index=False)
    d["budget_c"] = d.budget.where(d.budget <= max(budgets), 9999)  # agrupa los presupuestos completos (≈1 988–2 005) en una sola columna
    pd.set_option("display.width", 200)
    print("hipervolumen reentrenado de los 12 candidatos tras B evaluaciones (media de semillas; 9999 = presupuesto completo):\n",
          d.groupby(["budget_c", "arm"]).hv.mean().unstack().round(4).to_string())
    rng, out_rows = np.random.RandomState(0), []
    for B in list(budgets) + [9999]:
        w = d[d.budget_c == B].pivot(index="seed", columns="arm", values="hv").dropna()
        diff = (w["nsga3"] - w["rs"]).values
        ci = np.percentile([rng.choice(diff, len(diff)).mean() for _ in range(5000)], [2.5, 97.5])
        out_rows.append(dict(B=B, n=len(diff), media=diff.mean(), ic_lo=ci[0], ic_hi=ci[1], p=wilcoxon(diff, alternative="greater").pvalue if np.any(diff != 0) else 1.0, a_favor=int((diff > 0).sum())))
    c = pd.DataFrame(out_rows)
    pre = c.B.isin(budgets).values
    c.loc[pre, "p_holm"] = holm(c[pre].p.values)
    c.to_csv(Path(out) / "paired_contrasts.csv", index=False)
    print("\ncontrastes pareados nsga3 − rs por presupuesto (Wilcoxon unilateral; Holm sobre los presupuestos parciales):\n", c.round(4).to_string(index=False))
    ok = c[pre & (c.p_holm < 0.05) & (c.media > 0) & (c.a_favor >= 7)]
    print("\nLECTURA (preregistrada): " + (f"el MOEA es más eficiente en muestras (B = {ok.B.tolist()})." if len(ok) else "ningún presupuesto parcial cumple la regla: compatible con un espacio saturado también a presupuestos bajos."))


def analyze(out, seeds, k12_root):
    d = pd.concat([pd.read_csv(Path(out) / f"{a}_seed{s}.csv") for a in ARMS for s in seeds if (Path(out) / f"{a}_seed{s}.csv").exists()], ignore_index=True)
    hv = lambda g: _hv(np.stack([-g.acc.values, g["size"].values], 1))
    per = d.groupby(["arm", "seed"]).apply(hv).rename("hv_M").reset_index()
    for a in ARMS:  # referencia: el hipervolumen del hall of fame original de 12
        for s in seeds:
            ff = pd.read_csv(Path(k12_root) / a / f"seed{s}" / "final_front.csv"); ff = ff[ff.roles.str.split("|").map(lambda x: "hof" in x)]
            per.loc[(per.arm == a) & (per.seed == s), "hv_12"] = _hv(np.stack([-ff.test_acc_mean.values, ff.n_params_norm.values], 1))
    share = []
    for s in seeds:  # frente de la unión de ambos brazos: cuántos puntos aporta cada uno y cuánto supera su hipervolumen
        u = d[d.seed == s].drop_duplicates(["arm", "genome"]).reset_index(drop=True)
        front = u.iloc[_nd(np.stack([-u.acc.values, u["size"].values], 1))]
        share.append(dict(seed=s, hv_union=hv(u), **{f"frente_{a}": int((front.arm == a).sum()) for a in ARMS}))
    share = pd.DataFrame(share)
    per.to_csv(Path(out) / "per_run.csv", index=False)
    pd.set_option("display.width", 200)
    print(per.pivot(index="seed", columns="arm", values=["hv_12", "hv_M"]).round(4).to_string())
    print("\nmedia por brazo:\n", per.groupby("arm")[["hv_12", "hv_M"]].mean().round(4).to_string())
    print("\nfrente de la unión de ambos brazos (por semilla):\n", share.round(4).to_string(index=False), "\nmedia de puntos del frente por brazo:", {a: round(share[f"frente_{a}"].mean(), 2) for a in ARMS})
    rng, w = np.random.RandomState(0), per.pivot(index="seed", columns="arm", values="hv_M").dropna()
    diff = (w["nsga3"] - w["rs"]).values
    ci = np.percentile([rng.choice(diff, len(diff)).mean() for _ in range(5000)], [2.5, 97.5])
    p = wilcoxon(diff, alternative="greater").pvalue if np.any(diff != 0) else 1.0
    print(f"\nhipervolumen de los {int(d.groupby(['arm', 'seed']).size().mean())} candidatos reentrenados, nsga3 - rs: media {diff.mean():+.4f} IC95 [{ci[0]:+.4f}, {ci[1]:+.4f}], "
          f"p unilateral {p:.4f}, {int((diff > 0).sum())}/{len(diff)} semillas a favor\npor semilla: {np.round(diff, 4).tolist()}")
    print("LECTURA (preregistrada): " + ("C favorece a nsga3 (p < 0.05 y >= 8/10): el límite era la selección final de 12 con el proxy." if p < 0.05 and (diff > 0).sum() >= 8
                                         else "C no distingue los brazos: ver Pasos A/B para decidir entre fidelidad del proxy y espacio fácil."))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/moea2")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=0, help="solo pruebas: reentrena con menos rondas")
    ap.add_argument("--budgets", type=int, nargs="+", help="Paso D: reentrena los candidatos elegidos tras solo las primeras B evaluaciones de cada archivo")
    ap.add_argument("--rerank-dir", default="runs/rerank", help="Paso D: reentrenamientos del Paso C que se reutilizan")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    a.out = a.out or ("runs/rerank_prefix" if a.budgets else "runs/rerank")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.budgets:
        a.m = 12 if a.m == 24 else a.m  # Paso D usa K = 12, como la etapa final
        if not a.analyze_only:
            with Pool(a.workers) as p:
                for msg in p.imap_unordered(run_prefix_task, [(a.root, arm, s, a.budgets, a.m, a.out, a.threads, a.rounds, a.rerank_dir) for arm in a.arms for s in a.seeds]):
                    print(msg, flush=True)
        analyze_prefix(a.out, a.seeds, a.budgets, a.root)
        raise SystemExit
    if not a.analyze_only:
        with Pool(a.workers) as p:
            for msg in p.imap_unordered(run_task, [(a.root, arm, s, a.m, a.out, a.threads, a.rounds) for arm in a.arms for s in a.seeds]):
                print(msg, flush=True)
    analyze(a.out, a.seeds, a.root)
