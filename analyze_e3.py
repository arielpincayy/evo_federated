#!/usr/bin/env python
"""Agrega runs/e3/<método>/seed*/ y compara métodos con diferencias pareadas por semilla (descriptivo, n pequeño)."""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs/e3")
a = ap.parse_args()

rows, curves = [], []
for d in sorted(glob.glob(f"{a.root}/*/seed*")):
    m, seed = d.split("/")[-2], int(d.split("seed")[-1])
    if not os.path.exists(f"{d}/cost.json"):
        continue
    c = json.load(open(f"{d}/cost.json"))
    g = pd.read_csv(f"{d}/generation_metrics.csv")
    rows.append({"method": m, "seed": seed, "hv_test": c["hv_test_front"], "hv_monitor_final": g.hv_monitor.iloc[-1],
                 "hv_rep_final": g.hv_rep.iloc[-1], "est_rmse": g.est_rmse.mean(), "est_kendall": g.est_kendall.mean(),
                 "offspring_success": g.offspring_success.mean(), "diversity_last": g.diversity.iloc[-1],
                 "search_trainings": c["search_local_trainings"], "probe_trainings": c["probe_local_trainings"]})
    curves.append(g.assign(method=m, seed=seed))
per = pd.DataFrame(rows)
if per.empty:
    raise SystemExit("aún no hay corridas terminadas")
per.to_csv(f"{a.root}/per_run.csv", index=False)
pd.concat(curves).to_csv(f"{a.root}/curves.csv", index=False)
print(per.groupby("method")[["hv_test", "hv_monitor_final", "hv_rep_final", "est_rmse", "est_kendall", "offspring_success"]]
      .agg(["mean", "std", "count"]).round(4).to_string())
print("\ncoste de búsqueda (entrenamientos locales) por método:", per.groupby("method").search_trainings.mean().to_dict())

rng = np.random.RandomState(0)
out = []
for metric in ("hv_test", "hv_monitor_final"):
    piv = per.pivot(index="seed", columns="method", values=metric)
    for x, y in (("cs_nsga", "rand_nsga"), ("cs_nsga", "cs_rand"), ("rand_nsga", "rand_rand"), ("cs_rand", "rand_rand"), ("cs_nsga", "rand_rand"), ("rand_nsga", "cs_rand")):
        if x in piv and y in piv:
            d = (piv[x] - piv[y]).dropna().values
            if len(d):
                b = [rng.choice(d, len(d)).mean() for _ in range(5000)]
                out.append({"metric": metric, "contrast": f"{x} - {y}", "mean_diff": d.mean(), "ci_low": np.percentile(b, 2.5),
                            "ci_high": np.percentile(b, 97.5), "wins": int((d > 0).sum()), "n_seeds": len(d)})
res = pd.DataFrame(out)
res.to_csv(f"{a.root}/paired_contrasts.csv", index=False)
print("\nContrastes pareados (>0 favorece al primero):")
print(res.round(4).to_string(index=False))
