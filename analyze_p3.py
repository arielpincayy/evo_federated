#!/usr/bin/env python
"""Análisis de P3 (docs/P3_DESIGN.md): hipervolumen y contrastes pareados por semilla, desde runs/p3/{brazo}/seed{s}/final_front.csv y cost.json.
Uso: python analyze_p3.py [--root runs/p3] [--lam 1.0]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evofederated.evolution.cs_nas import _hv

ARMS, SEEDS, METRICS = ("p3", "rs", "base"), (0, 1, 2), ("hv", "best_S", "acc_best", "mean_size")


def metrics(d, lam):
    if "roles" in d:  # select_hof: solo los modelos del hall of fame (no la población inicial)
        d = d[d.roles.str.split("|").map(lambda r: "hof" in r)]
    d = d.groupby("genome").agg(acc=("test_acc_mean", "mean"), size=("n_params_norm", "first"))  # front_top: promedia las inicializaciones
    best = d.acc.idxmax()
    return dict(n_models=len(d), hv=_hv(np.stack([-d.acc.values, d["size"].values], 1)), best_S=float((d.acc - lam * d["size"]).max()), acc_best=float(d.acc.max()),
                size_of_best_acc=float(d["size"][best]), mean_size=float(d["size"].mean()), mean_acc=float(d.acc.mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/p3")
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    rows, missing = [], []
    for arm in ARMS:
        for s in SEEDS:
            f, c = Path(a.root) / arm / f"seed{s}" / "final_front.csv", Path(a.root) / arm / f"seed{s}" / "cost.json"
            if not (f.exists() and c.exists()):
                missing.append(f"{arm}/seed{s}")
                continue
            cost = json.load(open(c))
            rows.append(dict(arm=arm, seed=s, **metrics(pd.read_csv(f), a.lam), k=cost["k"], search_evals=cost["search_evals"], search_local_trainings=cost["search_local_trainings"],
                             elapsed_h=cost["elapsed_sec"] / 3600))
    d = pd.DataFrame(rows)
    if missing:
        print("sin terminar:", ", ".join(missing))
    if d.empty:
        raise SystemExit
    d.to_csv(Path(a.root) / "per_run.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.round(4).to_string(index=False))
    print("\nmedia por brazo:\n", d.groupby("arm")[list(METRICS) + ["size_of_best_acc", "k", "search_local_trainings", "elapsed_h"]].mean().round(4).to_string())
    rng, out = np.random.RandomState(0), []
    for other in ("base", "rs"):
        w = d[d.arm == "p3"].set_index("seed")[list(METRICS)].join(d[d.arm == other].set_index("seed")[list(METRICS)], rsuffix="_o", how="inner")
        for m in METRICS:
            diff = (w[m] - w[m + "_o"]).values
            if len(diff):
                ci = np.percentile([rng.choice(diff, len(diff)).mean() for _ in range(5000)], [2.5, 97.5])
                out.append(dict(contraste=f"p3 - {other}", metrica=m, n=len(diff), media=diff.mean(), ic_lo=ci[0], ic_hi=ci[1], por_semilla=" ".join(f"{x:+.4f}" for x in diff), ic_incluye_0=bool(ci[0] <= 0 <= ci[1])))
    c = pd.DataFrame(out)
    c.to_csv(Path(a.root) / "paired_contrasts.csv", index=False)
    print("\ncontrastes pareados (IC bootstrap descriptivo sobre las semillas):\n", c.round(4).to_string(index=False))
