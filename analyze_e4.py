#!/usr/bin/env python
"""Compara el HV en test del frente final antes (E3) y después del refinamiento multifidelidad (E4), pareado por semilla."""
import glob
import json

import numpy as np
import pandas as pd

rows = []
for d in sorted(glob.glob("runs/e3/*/seed*/refine/refine_cost.json")):
    m, s = d.split("/")[2], int(d.split("seed")[1].split("/")[0])
    c = json.load(open(d))
    rf = pd.read_csv(d.replace("refine_cost.json", "refined_candidates.csv"))
    ff = pd.read_csv(d.replace("refine_cost.json", "refined_final_front.csv"))
    orig = pd.read_csv(f"runs/e3/{m}/seed{s}/final_front.csv")
    rows.append({"method": m, "seed": s, "hv_original": c["hv_test_original"], "hv_refined": c["hv_test_refined"],
                 "best_test_acc_orig": orig.test_acc_mean.max(), "best_test_acc_ref": ff.test_acc_mean.max(),
                 "mean_test_acc_orig": orig.test_acc_mean.mean(), "mean_test_acc_ref": ff.test_acc_mean.mean(),
                 "refine_trainings": c["refine_local_trainings"], "n_front_ref": len(ff),
                 "same_genomes": len(set(orig.genome) & set(ff.genome))})
p = pd.DataFrame(rows)
p.to_csv("runs/e3/refine_per_run.csv", index=False)
print(p.groupby("method")[["hv_original", "hv_refined", "best_test_acc_orig", "best_test_acc_ref", "mean_test_acc_orig", "mean_test_acc_ref", "same_genomes"]].mean().round(4).to_string())
rng = np.random.RandomState(0)


def boot(d):
    b = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), int((d > 0).sum()), len(d)


out = []
for m, g in p.groupby("method"):  # H1: efecto del refinamiento dentro de cada método
    for metric in ("hv", "mean_test_acc", "best_test_acc"):
        d = (g[f"{metric}_refined" if metric == "hv" else f"{metric}_ref"] - g[f"{metric}_original" if metric == "hv" else f"{metric}_orig"]).values
        out.append({"contraste": f"H1 refinado − original ({m})", "métrica": metric, **dict(zip(["media", "ic_bajo", "ic_alto", "gana", "n"], boot(d)))})
piv_o = p.pivot(index="seed", columns="method", values="hv_original")
piv_r = p.pivot(index="seed", columns="method", values="hv_refined")
for tag, piv in (("original", piv_o), ("refinado", piv_r)):  # H2: diferencias entre métodos
    for a, b in (("cs_nsga", "rand_nsga"), ("cs_nsga", "cs_rand"), ("rand_nsga", "cs_rand")):
        out.append({"contraste": f"H2 {a} − {b} ({tag})", "métrica": "hv", **dict(zip(["media", "ic_bajo", "ic_alto", "gana", "n"], boot((piv[a] - piv[b]).values)))})
res = pd.DataFrame(out)
res.to_csv("runs/e3/refine_contrasts.csv", index=False)
print(res.round(4).to_string(index=False))
print("\nHV por semilla (original → refinado):")
print(p.pivot(index="seed", columns="method", values=["hv_original", "hv_refined"]).round(4).to_string())
