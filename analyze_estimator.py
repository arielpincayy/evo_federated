#!/usr/bin/env python
"""Agrega runs/estimator_e2/seed*/ y evalúa la regla E2 de docs/PROTOCOL.md.

Unidad de réplica = semilla de partición. Por semilla y variante: RMSE, sesgo y Kendall τ (dentro de cada sorteo).
Diferencias pareadas por semilla (variante - cluster_delta) con IC bootstrap al 95 % sobre semillas (n pequeño: descriptivo).
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs/estimator_e2")
ap.add_argument("--ref", default="cluster_delta")
a = ap.parse_args()

df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{a.root}/seed*/estimator_raw.csv")))
df["err"] = df.acc_rep - df.mon_acc_mean
rows = []
for (arm, seed), g in df.groupby(["arm", "seed"]):
    taus = [kendalltau(x.acc_rep, x.mon_acc_mean)[0] for _, x in g.groupby("generation") if x.acc_rep.std() > 0]
    rows.append({"arm": arm, "seed": seed, "bias": g.err.mean(), "rmse": np.sqrt((g.err ** 2).mean()),
                 "kendall": np.nanmean(taus), "acc_all_sd": g.mon_acc_mean.std(), "acc_all_mean": g.mon_acc_mean.mean()})
per = pd.DataFrame(rows)
per.to_csv(f"{a.root}/per_seed_summary.csv", index=False)
print(f"semillas: {sorted(per.seed.unique())}  evaluaciones por variante y semilla: {len(df) // per.groupby('arm').ngroups // per.seed.nunique()}")
print(per.groupby("arm")[["bias", "rmse", "kendall", "acc_all_mean", "acc_all_sd"]].agg(["mean", "std"]).round(4).to_string())

rng = np.random.RandomState(0)
out = []
for metric in ("rmse", "kendall"):
    piv = per.pivot(index="seed", columns="arm", values=metric)
    for arm in piv.columns:
        if arm == a.ref:
            continue
        d = (piv[arm] - piv[a.ref]).dropna().values  # >0: la variante tiene más rmse/kendall que la referencia
        boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        try:
            p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else float("nan")
        except ValueError:
            p = float("nan")
        out.append({"metric": metric, "arm_minus_ref": arm, "mean_diff": d.mean(), "ci_low": np.percentile(boots, 2.5),
                    "ci_high": np.percentile(boots, 97.5), "wilcoxon_p": p, "n_seeds": len(d)})
res = pd.DataFrame(out)
res.to_csv(f"{a.root}/paired_vs_{a.ref}.csv", index=False)
print(f"\nDiferencias pareadas frente a {a.ref} (rmse>0: peor que la referencia; kendall<0: peor):")
print(res.round(4).to_string(index=False))
